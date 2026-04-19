# Copyright (c) 2024, DeepLink. All rights reserved.
import importlib
import torch
from functools import lru_cache
from dlinfer.vendor import vendor_name

vendor = ["camb", "ascend"]


def fake_torch_compile(dynamic=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def pre_rms_norm(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pre rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance_q = (q * q).sum(-1, keepdim=True)
    variance_k = (k * k).sum(-1, keepdim=True)
    variance = torch.stack([variance_q, variance_k], dim=0)
    return variance


def post_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    variance: torch.Tensor,
    eps: float,
    embed_dim: int,
    dtype: torch.dtype,
):
    """Post rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance = variance / embed_dim + eps
    variance_q, variance_k = variance
    q = q * torch.rsqrt(variance_q)
    q = q.to(dtype) * weight_q
    k = k * torch.rsqrt(variance_k)
    k = k.to(dtype) * weight_k
    return q, k


def patch_compiled_func():
    import torch

    real_torch_compile = torch.compile
    torch.compile = fake_torch_compile
    from lmdeploy.pytorch.models import internvl, internvl3_hf

    internvl.pre_rms_norm = pre_rms_norm
    internvl.post_rms_norm = post_rms_norm
    internvl3_hf.pre_rms_norm = pre_rms_norm
    internvl3_hf.post_rms_norm = post_rms_norm
    torch.compile = real_torch_compile


def patch_async_sampling_logits():
    from torch.profiler import record_function
    from lmdeploy.pytorch.engine.model_agent import BaseModelAgent
    from lmdeploy.pytorch.engine.model_agent.agent import BatchedLogProbs
    from lmdeploy.pytorch.engine.logits_process import (
        SamplingInputs,
        FusedLogitsProcessor,
    )

    async def async_sampling_logits(
        self, logits: torch.Tensor, sampling_inputs: SamplingInputs
    ):
        """Sampling logits."""
        # record function does not support async function
        # so we can not decorate it on async_sampling_logits
        with record_function("sampling_logits"):
            logits = logits.to(torch.float32)
            logits_processor = FusedLogitsProcessor(
                sampling_inputs,
                logprobs_mode=self.misc_config.logprobs_mode,
                guided_decoding_manager=self.guided_decoding_manager,
            )
            origin_logits = logits
            logits, raw_logprobs = await logits_processor(origin_logits)
            next_token_ids = logits_processor.sampling(logits)
            logprobs = logits_processor.compute_logprobs(raw_logprobs, next_token_ids)
            if logprobs is not None:
                logprobs = BatchedLogProbs(
                    vals=logprobs[0],
                    indices=logprobs[1],
                )

        return next_token_ids, logprobs

    BaseModelAgent.async_sampling_logits = async_sampling_logits


def patch_rejection_sampler():
    from lmdeploy.pytorch.spec_decode import reject_sampler as _reject_sampler_mod
    _orig_rejection_sample = _reject_sampler_mod.rejection_sample

    def _patched_rejection_sample(
        target_logits,
        draft_token_ids,
        bonus_token_ids,
        sampling_inputs,
        draft_probs=None,
    ):
        if sampling_inputs.max_top_k == 1:
            return _orig_rejection_sample(
                target_logits,
                draft_token_ids,
                bonus_token_ids,
                sampling_inputs,
                draft_probs=draft_probs,
            )

        assert draft_probs is None or draft_probs.is_contiguous()
        if not draft_token_ids.is_contiguous():
            draft_token_ids = draft_token_ids.contiguous()

        if not target_logits.is_contiguous():
            target_logits = target_logits.contiguous()

        batch_size, num_spec_tokens = draft_token_ids.shape
        device = target_logits.device

        output_token_ids = torch.full(
            (batch_size, num_spec_tokens + 1),
            _reject_sampler_mod.PLACEHOLDER_TOKEN_ID,
            dtype=torch.long,
            device=device,
        )

        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        if sampling_inputs.top_k is not None:
            is_greedy = (sampling_inputs.top_k == 1)
            if not torch.is_tensor(is_greedy):
                is_greedy = torch.full(
                    (batch_size,), bool(is_greedy), dtype=torch.bool, device=device
                )
            else:
                is_greedy = is_greedy.to(device=device, dtype=torch.bool)
        else:
            is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)

        target_argmax = target_probs.argmax(dim=-1)
        uniform_probs = torch.rand(
            (batch_size, num_spec_tokens), dtype=torch.float64, device=device
        )
        inv_q = torch.empty(
            (batch_size, target_probs.shape[-1]), dtype=torch.float32, device=device
        )
        inv_q.exponential_()
        inv_q = inv_q.reciprocal()

        recovered_token_ids = torch.empty(
            (batch_size, num_spec_tokens), dtype=torch.long, device=device
        )
        zero = target_probs.new_tensor(0.0)
        for batch_idx in range(batch_size):
            if bool(is_greedy[batch_idx].item()):
                continue
            batch_inv_q = inv_q[batch_idx]
            for pos in range(num_spec_tokens):
                draft_token_id = draft_token_ids[batch_idx, pos]
                if draft_probs is None:
                    prob = target_probs[batch_idx, pos].clone()
                    prob[draft_token_id] = 0.0
                else:
                    prob = torch.maximum(
                        target_probs[batch_idx, pos] - draft_probs[batch_idx, pos],
                        zero,
                    )
                recovered_token_ids[batch_idx, pos] = torch.argmax(prob * batch_inv_q)

        for batch_idx in range(batch_size):
            rejected = False
            if bool(is_greedy[batch_idx].item()):
                for pos in range(num_spec_tokens):
                    token_id = target_argmax[batch_idx, pos]
                    output_token_ids[batch_idx, pos] = token_id
                    if draft_token_ids[batch_idx, pos] != token_id:
                        rejected = True
                        break
            else:
                for pos in range(num_spec_tokens):
                    draft_token_id = draft_token_ids[batch_idx, pos]
                    if draft_probs is None:
                        draft_prob = 1.0
                    else:
                        draft_prob = float(
                            draft_probs[batch_idx, pos, draft_token_id].item()
                        )
                    target_prob = float(
                        target_probs[batch_idx, pos, draft_token_id].item()
                    )
                    uniform_prob = float(uniform_probs[batch_idx, pos].item())
                    if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                        token_id = draft_token_id
                    else:
                        token_id = recovered_token_ids[batch_idx, pos]
                        rejected = True
                    output_token_ids[batch_idx, pos] = token_id
                    if rejected:
                        break

            if not rejected:
                output_token_ids[batch_idx, num_spec_tokens] = bonus_token_ids[batch_idx]

        return _reject_sampler_mod._extract_outputs(output_token_ids, num_spec_tokens)

    _reject_sampler_mod.rejection_sample = _patched_rejection_sample


def patch_spec_decode_runtime():
    from torch.profiler import record_function
    from lmdeploy.pytorch.engine.model_agent import BaseModelAgent
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    _orig_build_cache_engine = BaseModelAgent.build_cache_engine
    _orig_forward_impl = BaseModelAgent._forward_impl

    def _is_ascend_agent(agent):
        return getattr(getattr(agent, "backend_config", None), "device_type", None) == "ascend"

    def _set_main_runtime(self, model, cache_engine, state_cache_engine, stream):
        self.main_model = model
        self.main_cache_engine = cache_engine
        self.main_state_cache_engine = state_cache_engine
        self.main_stream = stream

    def _maybe_snapshot_main_states(self, model_inputs):
        self._main_state_snapshot = None
        self._main_replay_inputs = None
        state_cache_engine = getattr(self, "main_state_cache_engine", None)
        main_model = getattr(self, "main_model", None)
        if state_cache_engine is None or main_model is None:
            return
        if not model_inputs.is_decoding or model_inputs.max_q_seqlen <= 1:
            return
        mem_pool = getattr(state_cache_engine, "mem_pool", None)
        if mem_pool is None or mem_pool.numel() == 0:
            return
        self._main_state_snapshot = mem_pool.clone()
        self._main_replay_inputs = model_inputs.clone()

    def _build_replay_inputs(self, model_inputs, output_token_ids):
        valid_mask = output_token_ids.ge(0)
        seq_length = valid_mask.sum(dim=-1).to(model_inputs.seq_length.dtype)
        if torch.any(seq_length <= 0):
            return None

        replay_ids = [row[mask] for row, mask in zip(output_token_ids, valid_mask)]
        input_ids = torch.cat(replay_ids, dim=0).unsqueeze(0)

        mrope_pos_ids = model_inputs.mrope_pos_ids
        if mrope_pos_ids is not None:
            mrope_chunks = []
            reshaped = mrope_pos_ids.unflatten(1, (-1, model_inputs.max_q_seqlen))
            for batch_idx, replay_len in enumerate(seq_length.tolist()):
                mrope_chunks.append(reshaped[:, batch_idx, :replay_len])
            mrope_pos_ids = torch.cat(mrope_chunks, dim=1)

        max_q_seqlen = int(seq_length.max().item())
        max_kv_seqlen = int((model_inputs.history_lengths + seq_length).max().item())
        sum_kv_seqlen = int((model_inputs.history_lengths + seq_length).sum().item())
        return model_inputs.clone(
            input_ids=input_ids,
            seq_length=seq_length,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_kv_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            target_hidden_states=None,
            target_position_ids=None,
            target_inputs_embeds=None,
            mrope_pos_ids=mrope_pos_ids,
        )

    def _maybe_replay_main_states(self, extra_inputs):
        state_snapshot = getattr(self, "_main_state_snapshot", None)
        replay_template = getattr(self, "_main_replay_inputs", None)
        if state_snapshot is None or replay_template is None:
            return
        try:
            if extra_inputs.num_rejected_tokens is None or not torch.any(extra_inputs.num_rejected_tokens > 0):
                return
            replay_inputs = self._build_replay_inputs(replay_template, extra_inputs.output_token_ids)
            if replay_inputs is None:
                return

            main_model = getattr(self, "main_model", None)
            main_cache_engine = getattr(self, "main_cache_engine", None)
            state_cache_engine = getattr(self, "main_state_cache_engine", None)
            if main_model is None or main_cache_engine is None or state_cache_engine is None:
                return

            state_cache_engine.mem_pool.copy_(state_snapshot)
            from lmdeploy.pytorch.engine.model_agent.agent import model_forward as _main_model_forward

            _main_model_forward(
                main_model,
                replay_inputs,
                main_cache_engine,
                state_cache_engine,
                stream=getattr(self, "main_stream", None),
            )
        finally:
            self._main_state_snapshot = None
            self._main_replay_inputs = None

    def _patched_build_cache_engine(self):
        if _is_ascend_agent(self):
            state_shapes = getattr(self.model_config, "states_shapes", [])
            self.cache_config.states_shapes = state_shapes
            if self.cache_config.num_state_caches is None and len(state_shapes) > 0:
                self.cache_config.num_state_caches = int(self.cache_config.max_batches + 1)

        _orig_build_cache_engine(self)

        if (
            _is_ascend_agent(self)
            and self.spec_agent is not None
            and self.spec_agent.is_enabled()
        ):
            self.spec_agent.set_main_runtime(
                self.patched_model,
                self.cache_engine,
                self.state_cache_engine,
                self.stream,
            )

    def _patched_forward_impl(self, inputs):
        if (
            _is_ascend_agent(self)
            and self.spec_agent is not None
            and self.spec_agent.is_enabled()
        ):
            self.spec_agent.maybe_snapshot_main_states(inputs)
        return _orig_forward_impl(self, inputs)

    async def _patched_async_model_forward(
        self,
        model_inputs,
        extra_inputs,
        sampling_inputs,
    ):
        with record_function("spec_rejection_sampling"):
            draft_extra_inputs = await self._rejection_sampling(
                model_inputs, extra_inputs, sampling_inputs
            )
        self._maybe_replay_main_states(draft_extra_inputs)
        draft_model_inputs, draft_extra_inputs = self._prepare_inputs_from_main(
            model_inputs, draft_extra_inputs
        )
        return await self._async_model_forward(
            draft_model_inputs, draft_extra_inputs, sampling_inputs
        )

    BaseModelAgent.build_cache_engine = _patched_build_cache_engine
    BaseModelAgent._forward_impl = _patched_forward_impl
    SpecModelAgent.set_main_runtime = _set_main_runtime
    SpecModelAgent.maybe_snapshot_main_states = _maybe_snapshot_main_states
    SpecModelAgent._build_replay_inputs = _build_replay_inputs
    SpecModelAgent._maybe_replay_main_states = _maybe_replay_main_states
    SpecModelAgent.async_model_forward = _patched_async_model_forward


##### patch cache engine #####
def patch_contiguous_cache_engine():
    from lmdeploy.pytorch.config import CacheConfig, ModelConfig
    from functools import reduce
    from math import gcd
    from lmdeploy.pytorch.engine import cache_engine

    @classmethod
    def _cache_engine_allocate_caches(
        cls,
        num_blocks: int,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        world_size: int,
        device: str,
    ):
        """Allocate caches."""
        num_layers = model_config.num_layers

        # get all descs
        k_cache_desc = cls.get_k_cache_desc(model_config, cache_config, world_size)
        v_cache_desc = cls.get_v_cache_desc(model_config, cache_config, world_size)
        quant_cache_descs = cls.get_quant_cache_descs(
            k_cache_desc, v_cache_desc, model_config, cache_config
        )
        custom_cache_descs = cls.get_custom_cache_descs(model_config, cache_config)
        cache_descs = (
            [k_cache_desc, v_cache_desc] + quant_cache_descs + custom_cache_descs
        )

        # get mempool size
        mem_pool_size = 0
        alignments = []
        for desc in cache_descs:
            mem_pool_size += desc.aligned_size
            alignments.append(desc.alignment)

        # compute gcd of alignments
        alignments_gcd = reduce(gcd, alignments) if alignments else 1
        assert (
            mem_pool_size % alignments_gcd == 0
        ), "mem_pool_size must be divisible by alignments_gcd"

        # create pool
        mem_pool = torch.zeros(
            (mem_pool_size // alignments_gcd, num_layers, num_blocks, alignments_gcd),
            dtype=torch.uint8,
            device=device,
        )

        # slice caches
        caches = []
        remain_pool = mem_pool
        for desc in cache_descs:
            cache = (
                remain_pool[: desc.size // alignments_gcd, :, :, :]
                .view(desc.dtype)
                .view((num_layers, num_blocks, *desc.shape))
            )
            remain_pool = remain_pool[desc.aligned_size // alignments_gcd :, :, :, :]
            caches.append(cache)
        return mem_pool, caches

    cache_engine.CacheEngine.allocate_caches = _cache_engine_allocate_caches


##### patch state cache engine #####
def patch_state_cache_engine():
    from typing import List, Tuple
    from lmdeploy.pytorch.engine import cache_engine
    from lmdeploy.pytorch.engine.cache_engine import CacheDesc

    @staticmethod
    def _state_cache_engine_allocate_caches(
        num_caches: int,
        state_shapes: List[Tuple[Tuple[int], torch.dtype]],
        device: torch.device,
    ):
        """Allocate cache implement.

        Each state is allocated as an independent contiguous tensor of shape
        (num_caches, *shape).  A single shared pool of shape
        (num_caches, total_pool_size) would give views with stride[0] ==
        total_pool_size instead of the per-state numel, making every slice
        non-contiguous and breaking NPU ops that require contiguous input.
        """

        cache_dtype = torch.int8
        if len(state_shapes) == 0 or num_caches == 0:
            return torch.empty((0, 0), dtype=cache_dtype, device=device), []

        cache_descs = [CacheDesc(shape, dtype) for shape, dtype in state_shapes]

        # Allocate each state as a separate contiguous tensor.
        caches = []
        for desc in cache_descs:
            cache = torch.zeros(
                (num_caches, *desc.shape), dtype=desc.dtype, device=device
            )
            caches.append(cache)

        # mem_pool is used by two callers:
        #   1. get_cache_state_size(): always calls with device='meta' to compute byte
        #      counts — the tensor is never materialised on a real device.
        #   2. init_caches(): patched below to zero individual caches directly, so it
        #      no longer touches mem_pool at all.
        # Therefore we only need a correctly-sized pool on 'meta'; for real devices we
        # return an empty placeholder to avoid doubling the state-cache memory footprint.
        total_bytes = sum(desc.aligned_size for desc in cache_descs)
        if str(device) == "meta":
            mem_pool = torch.empty(
                (num_caches, total_bytes), dtype=cache_dtype, device=device
            )
        else:
            mem_pool = torch.empty(0, dtype=cache_dtype, device=device)
        return mem_pool, caches

    def _state_cache_engine_init_caches(self, idx: torch.Tensor, mask: torch.Tensor):
        """Initialize state caches by zeroing each individual cache tensor."""
        if idx is None:
            return
        if len(self._state_caches) <= 0:
            return
        num_caches = self.cache_config.num_state_caches
        cache_masks = torch.zeros((num_caches,), dtype=torch.bool, device=idx.device)
        cache_masks.index_copy_(0, idx, mask)
        for cache in self._state_caches:
            reshaped_mask = cache_masks.view((-1,) + (1,) * (cache.dim() - 1))
            cache.masked_fill_(reshaped_mask, 0)

    cache_engine.StateCacheEngine.allocate_caches = _state_cache_engine_allocate_caches
    cache_engine.StateCacheEngine.init_caches = _state_cache_engine_init_caches


def patch_gated_delta_net():
    import torch.nn.functional as F
    from typing import Any, Sequence, Tuple
    from torch.profiler import record_function

    from lmdeploy.pytorch.nn import gated_delta
    from lmdeploy.pytorch.nn.gated_delta import GatedDeltaMeta
    from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

    from dlinfer.vendor.ascend.triton_ops import RMSNormGated
    from dlinfer.vendor.ascend.triton_ops import (
        causal_conv1d_fn,
        causal_conv1d_update_npu,
    )
    from dlinfer.vendor.ascend.triton_ops import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
        fused_sigmoid_gating_delta_rule_update,
    )

    class AscendGatedDeltaMeta:

        def __init__(
            self,
            num_tokens: int,
            conv_kernel_size: int,
            state_ids: torch.Tensor,
            attn_metadata: Any,
        ):
            self.is_multi_token_decoding = getattr(attn_metadata, 'is_multi_token_decoding', False)
            # Keep decode semantics for linear-attention state updates even when
            # full attention uses a prefill-style TND verify path.
            self.is_decoding = attn_metadata.is_decoding or self.is_multi_token_decoding
            self.cu_seqlens = attn_metadata.q_start_loc
            self.num_spec_tokens = get_step_ctx_manager().build_ctx.num_spec_tokens

            query_lens = None
            num_seqs = 1
            if self.cu_seqlens is not None:
                query_lens = torch.diff(self.cu_seqlens).to(torch.int32)
                num_seqs = max(int(self.cu_seqlens.numel()) - 1, 1)
            self.max_query_len = max(num_tokens // num_seqs, 1)
            self.cache_seqlens = None
            self.spec_state_offsets = None
            kv_seqlens_device = getattr(attn_metadata, 'kv_seqlens_device', None)
            if query_lens is not None and kv_seqlens_device is not None:
                kv_seqlens = kv_seqlens_device.to(dtype=torch.int32)
                self.cache_seqlens = (kv_seqlens - query_lens).contiguous()
                if self.num_spec_tokens > 0 and not self.is_decoding:
                    state_slots = 1 + self.num_spec_tokens
                    self.spec_state_offsets = (
                        torch.remainder(self.cache_seqlens, state_slots),
                        torch.remainder(kv_seqlens, state_slots),
                    )
            self.num_accepted_tokens = getattr(attn_metadata, 'num_accepted_tokens', None)
            if self.num_accepted_tokens is None and self.is_multi_token_decoding and query_lens is not None:
                self.num_accepted_tokens = torch.ones(query_lens.size(0), dtype=torch.int32, device=self.cu_seqlens.device)
            elif self.num_accepted_tokens is not None:
                self.num_accepted_tokens = self.num_accepted_tokens.to(
                    device=self.cu_seqlens.device if self.cu_seqlens is not None else state_ids.device,
                    dtype=torch.int32,
                ).contiguous()

            # state_ids, fill invalid state with 0
            self.state_ids = state_ids.clamp(0)
            self.has_initial_state = attn_metadata.has_initial_state
            self.conv_state_indices = self.state_ids.to(torch.int32)

    def build_rmsnorm_gated(hidden_size: int, eps=1e-6, **kwargs):
        device = kwargs["device"]
        return RMSNormGated(hidden_size, eps=eps, norm_before_gate=True, device=device)

    class AscendCausalConv1dFunc:

        def __init__(self, activation: str = "silu"):
            self.causal_conv1d_fn = causal_conv1d_fn
            self.causal_conv1d_update = causal_conv1d_update_npu
            self.activation = activation

        def conv1d_func(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            conv_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            """
            x: (b, seqlen, dim)
            seqlen: (b)
            out: (b, seqlen, dim)
            conv_state: (b, dim, kernel_size)
            """
            out = self.causal_conv1d_fn(
                x.t(),
                weight,
                bias,
                activation=self.activation,
                conv_states=conv_state.transpose(1, 2),
                has_initial_state=gated_delta_meta.has_initial_state,
                cache_indices=gated_delta_meta.conv_state_indices,
                query_start_loc=gated_delta_meta.cu_seqlens,
            )

            out = out.t().unsqueeze(0)

            return out, conv_state

        # 替换
        def conv1d_update(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            conv_state: torch.Tensor,
            conv_state_indices: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            update_kwargs = {}
            validate_data = True
            if getattr(gated_delta_meta, 'is_multi_token_decoding', False):
                update_kwargs.update(
                    num_accepted_tokens=gated_delta_meta.num_accepted_tokens,
                    query_start_loc=gated_delta_meta.cu_seqlens,
                    max_query_len=gated_delta_meta.max_query_len,
                )
                validate_data = False
            out = self.causal_conv1d_update(
                x,
                conv_state,
                weight.t().contiguous(),
                bias,
                self.activation,
                conv_state_indices=conv_state_indices,
                validate_data=validate_data,
                **update_kwargs,
            )
            return out.unsqueeze(0), conv_state

        @record_function("causal_conv1d")
        def __call__(
            self,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            conv_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            weight_reshaped = weight.squeeze(1)
            x = x.squeeze(0)

            if gated_delta_meta.is_decoding:
                conv_state_indices = gated_delta_meta.conv_state_indices
                return self.conv1d_update(
                    x, weight_reshaped, bias, conv_state, conv_state_indices, gated_delta_meta
                )
            return self.conv1d_func(
                x, weight_reshaped, bias, conv_state, gated_delta_meta=gated_delta_meta
            )

    class AscendGatedDelta:

        def __init__(self, use_qk_l2norm_in_kernel: bool = True):
            self.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
            self.fused_sigmoid_gating_delta_rule_update = (
                fused_sigmoid_gating_delta_rule_update
            )
            self.chunk_gated_delta_rule = chunk_gated_delta_rule
            self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

        @staticmethod
        def _get_decode_state_indices(
            state_ids: torch.Tensor,
            cache_seqlens: torch.Tensor,
            num_slots: int,
            query_len: int,
        ) -> torch.Tensor:
            """Map LMDeploy's per-sequence slot layout to per-token state ids."""
            token_offsets = torch.arange(
                query_len,
                device=cache_seqlens.device,
                dtype=torch.int64,
            )
            slot_offsets = torch.remainder(
                cache_seqlens.to(torch.int64)[:, None] + token_offsets[None],
                num_slots,
            )
            return (state_ids.to(torch.int64)[:, None] * num_slots + slot_offsets).contiguous()

        def __call__(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            A_log: torch.Tensor,
            dt_bias: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
            recurrent_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
        ):
            """call."""

            is_decoding = gated_delta_meta.is_decoding
            is_multi_token_decode = getattr(gated_delta_meta, 'is_multi_token_decoding', False)
            beta = b.sigmoid()
            # If the model is loaded in fp16, without the .float() here, A might be -inf
            g = (-A_log.float().exp()) * F.softplus(a.float() + dt_bias)

            if is_decoding:
                indices = gated_delta_meta.state_ids
                cu_seqlens = gated_delta_meta.cu_seqlens
                if is_multi_token_decode:
                    query_len = gated_delta_meta.max_query_len
                    state_slots = recurrent_state.size(1)
                    flat_recurrent_state = recurrent_state.view(-1, *recurrent_state.shape[2:])
                    state_indices = self._get_decode_state_indices(
                        indices,
                        gated_delta_meta.cache_seqlens,
                        state_slots,
                        query_len,
                    )
                    core_attn_out, _ = self.fused_recurrent_gated_delta_rule(
                        q=query.contiguous(),
                        k=key.contiguous(),
                        v=value.contiguous(),
                        g=g.contiguous(),
                        beta=beta.contiguous(),
                        initial_state=flat_recurrent_state,
                        inplace_final_state=True,
                        cu_seqlens=cu_seqlens,
                        ssm_state_indices=state_indices,
                        num_accepted_tokens=gated_delta_meta.num_accepted_tokens,
                        use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                    )
                    return core_attn_out, None

                # Single-token decode: use the optimized update kernel
                initial_state_source = recurrent_state
                initial_state_indices = indices
                if recurrent_state.dim() == 5:
                    state_slots = recurrent_state.size(1)
                    flat_recurrent_state = recurrent_state.view(-1, *recurrent_state.shape[2:])
                    slot_offsets = torch.remainder(
                        gated_delta_meta.cache_seqlens.to(torch.int64),
                        state_slots,
                    )
                    initial_state_source = flat_recurrent_state
                    initial_state_indices = (
                        indices.to(torch.int64) * state_slots + slot_offsets
                    ).contiguous()
                core_attn_out = self.fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    dt_bias=dt_bias,
                    q=query,
                    k=key,
                    v=value.contiguous(),
                    a=a.contiguous(),
                    b=b.contiguous(),
                    initial_state_source=initial_state_source,
                    initial_state_indices=initial_state_indices,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                )
                last_recurrent_state = None
            else:
                if gated_delta_meta.spec_state_offsets is not None:
                    read_offsets, write_offsets = gated_delta_meta.spec_state_offsets
                    initial_state = recurrent_state[gated_delta_meta.state_ids, read_offsets]
                else:
                    initial_state = recurrent_state[gated_delta_meta.state_ids]
                initial_state[~gated_delta_meta.has_initial_state, ...] = 0
                core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                    q=query,
                    k=key,
                    v=value.contiguous(),
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=gated_delta_meta.cu_seqlens,
                    head_first=False,
                    use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                )
                if gated_delta_meta.spec_state_offsets is not None:
                    recurrent_state[gated_delta_meta.state_ids, write_offsets] = last_recurrent_state.to(
                        recurrent_state.dtype
                    )
                else:
                    recurrent_state[gated_delta_meta.state_ids] = last_recurrent_state.to(
                        recurrent_state.dtype
                    )

            return core_attn_out, last_recurrent_state

    gated_delta.GatedDeltaMeta = AscendGatedDeltaMeta
    gated_delta.CausalConv1dFunc = AscendCausalConv1dFunc
    gated_delta.GatedDelta = AscendGatedDelta
    gated_delta.build_rmsnorm_gated = build_rmsnorm_gated


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    if vendor_name_str in vendor:
        importlib.import_module(f".{vendor_name_str}", __package__)


def patch_attention_is_tp():
    """Monkey-patch Qwen3_5Attention to skip TP head division for draft model.

    The MTP draft model uses is_tp=False to keep full head counts on each
    rank. Qwen3_5Attention already passes is_tp to build_qkv_proj and
    build_o_proj, but Attention.__init__ always calls _update_num_heads.
    We temporarily replace _update_num_heads with an identity function
    during Qwen3_5Attention.__init__ when is_tp=False.
    """
    from lmdeploy.pytorch.nn import attention as _attn_mod
    from lmdeploy.pytorch.models import qwen3_5

    _orig_update = _attn_mod._update_num_heads
    _identity_update = lambda nh, nkv: (nh, nkv)
    _orig_init = qwen3_5.Qwen3_5Attention.__init__

    def _patched_init(self, config, layer_idx, dtype=None, device=None,
                      prefix='', is_tp=True):
        if not is_tp:
            _attn_mod._update_num_heads = _identity_update
        try:
            _orig_init(self, config, layer_idx, dtype=dtype, device=device,
                       prefix=prefix, is_tp=is_tp)
        finally:
            if not is_tp:
                _attn_mod._update_num_heads = _orig_update

    qwen3_5.Qwen3_5Attention.__init__ = _patched_init


def patch_qwen3_5():
    import torch
    from typing import List

    from lmdeploy.utils import is_bf16_supported
    from lmdeploy.pytorch.configurations.default import DefaultModelConfigBuilder
    from lmdeploy.pytorch.configurations.qwen3_next import _check_env_qwen3_next
    from lmdeploy.vl.constants import Modality

    from lmdeploy.pytorch.nn.gated_delta import GatedDeltaMeta, CausalConv1d
    from lmdeploy.pytorch.model_inputs import StepContext
    from lmdeploy.pytorch.configurations.qwen3_5 import Qwen3_5ModelConfigBuilder
    from lmdeploy.pytorch.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5GatedDeltaNet,
    )

    @classmethod
    def custom_build(cls, hf_config, model_path: str = None, tp: int = 1, **kwargs):
        """build."""
        is_draft_model = kwargs.get("is_draft_model", False)
        spec_method = kwargs.get("spec_method", None)
        num_spec_tokens = kwargs.get("num_spec_tokens", 0)
        text_config = hf_config.text_config
        # propagate quantization_config from top-level hf_config into text_config
        quantization_config = getattr(hf_config, "quantization_config", None)
        if quantization_config is not None and not hasattr(
            text_config, "quantization_config"
        ):
            text_config.quantization_config = quantization_config
        cfg = DefaultModelConfigBuilder.build(text_config, model_path, tp=tp, **kwargs)

        # update num layers
        num_layers = cfg.num_layers
        layer_types = text_config.layer_types
        num_delta_layers = sum([1 for lt in layer_types if lt == "linear_attention"])
        num_full_layers = num_layers - num_delta_layers
        cfg.num_layers = num_full_layers

        # set state shapes
        head_k_dim = text_config.linear_key_head_dim
        head_v_dim = text_config.linear_value_head_dim
        num_v_heads = text_config.linear_num_value_heads // tp
        num_k_heads = text_config.linear_num_key_heads // tp
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        conv_dim = key_dim * 2 + value_dim
        conv_kernel_size = text_config.linear_conv_kernel_dim + num_spec_tokens

        # Ascend Patch
        conv_state_shape = (conv_kernel_size, conv_dim)
        if num_spec_tokens > 0:
            recurrent_state_shape = (1 + num_spec_tokens, num_v_heads, head_k_dim, head_v_dim)
        else:
            recurrent_state_shape = (num_v_heads, head_k_dim, head_v_dim)

        device_type = kwargs.get("device_type", "cuda")
        if is_bf16_supported(device_type):
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        # Ascend Patch
        # Use per-layer shapes so each cache slice is (num_caches, ...) — contiguous by
        # construction. Storing num_delta_layers as the first shape dim would require a
        # transpose later, producing non-contiguous views.
        cfg.states_shapes = [(conv_state_shape, dtype)] * num_delta_layers + [
            (recurrent_state_shape, torch.float32)
        ] * num_delta_layers

        cfg.is_gated_delta = True
        cfg.check_env_func = _check_env_qwen3_next

        cfg.use_mrope = True

        # Speculative decoding support
        if spec_method is not None:
            assert spec_method == "qwen3_5_mtp"
            cfg.model_paradigm = "ar_spec"

        if is_draft_model:
            hf_config.architectures = ["Qwen3_5MTPModel"]
            if getattr(hf_config, "auto_map", None):
                hf_config.auto_map = {}
            cfg.model_paradigm = "ar_spec"
            cfg.num_layers = text_config.mtp_num_hidden_layers
            cfg.states_shapes = []
            # Draft model uses is_tp=False — each rank runs the full model
            # independently, so keep the replicated KV head count for correct
            # cache allocation.

        return cfg

    def custom_prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
    ):
        """Prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # Ascend Patch
        # make past_key_values
        # state_caches holds num_delta_layers conv entries then num_delta_layers recurrent
        # entries, each already shaped (num_caches, ...) and contiguous.
        n = len(context.state_caches) // 2
        state_caches = list(zip(context.state_caches[:n], context.state_caches[n:]))

        past_key_values = list(past_key_values)
        new_past_key_values = []
        for layer_type in self.config.text_config.layer_types:
            if layer_type == "linear_attention":
                new_past_key_values.append(state_caches.pop(0))
            elif layer_type == "full_attention":
                new_past_key_values.append(past_key_values.pop(0))

        # vlm inputs
        pixel_values = None
        vis_cu_seqlens = None
        vis_pos_emb = None
        image_mask = None
        grid_thw = None
        pos_embeds = None
        if context.input_multimodals is not None:
            mm_inputs = [
                input_mm.get("mm_data", []) for input_mm in context.input_multimodals
            ]
            # flatten batch
            mm_inputs = [item for sublist in mm_inputs for item in sublist]

            if len(mm_inputs) > 0:
                modality = mm_inputs[0].modality
                pixel_values = torch.cat([inp.data for inp in mm_inputs])

                image_token_id = mm_inputs[0].meta.get("image_token_id")
                video_token_id = mm_inputs[0].meta.get("video_token_id")
                mm_token_id = (
                    image_token_id if modality == Modality.IMAGE else video_token_id
                )
                image_mask = input_ids == mm_token_id

                grid_thw = torch.cat(
                    [data.meta["grid_thw"] for data in mm_inputs]
                ).cpu()
                vis_pos_emb = self.model.visual.rot_pos_emb(grid_thw)
                pos_embeds = self.model.visual.fast_pos_embed_interpolate(grid_thw)
                vis_cu_seqlens = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
                ).to(pixel_values.device)
                vis_cu_seqlens = vis_cu_seqlens.cumsum(dim=0, dtype=torch.int32)
                vis_pos_emb = vis_pos_emb.repeat(1, 2)
                vis_pos_emb = (vis_pos_emb.cos(), vis_pos_emb.sin())

        mrope_position_ids = getattr(context, "mrope_position_ids", None)

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(
                inputs_embeds
            )

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=context.state_offsets,
            # vl inputs
            mrope_position_ids=mrope_position_ids,
            pixel_values=pixel_values,
            vis_cu_seqlens=vis_cu_seqlens,
            vis_pos_emb=vis_pos_emb,
            image_mask=image_mask,
            grid_thw=grid_thw,
            pos_embeds=pos_embeds,
        )

    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
        gated_delta_meta: GatedDeltaMeta,
    ):
        """forward."""

        # load states
        conv_state, recurrent_state = self._load_state(past_key_value, gated_delta_meta)

        # inputs proj
        projected_states_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        # [..., ng, np/ng * hn] -> [..., np, hn]
        z = z.unflatten(-1, (-1, self.head_v_dim))
        projected_states_ba = self.in_proj_ba(hidden_states)
        b, a = self.fix_ba_ordering(projected_states_ba)

        mixed_qkv = projected_states_qkv
        mixed_qkv, conv_state = self.conv1d(
            mixed_qkv, conv_state, gated_delta_meta=gated_delta_meta
        )

        tp = (self.key_dim * 2 + self.value_dim) // mixed_qkv.size(-1)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // tp,
                self.key_dim // tp,
                self.value_dim // tp,
            ],
            dim=-1,
        )
        query = query.unflatten(-1, (-1, self.head_k_dim))
        key = key.unflatten(-1, (-1, self.head_k_dim))
        value = value.unflatten(-1, (-1, self.head_v_dim))

        # beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        # g = self.get_A_log_exp() * F.softplus(a.float() + self.dt_bias)
        if self.kv_ratio > 1:
            query = query.repeat_interleave(self.kv_ratio, dim=-2)
            key = key.repeat_interleave(self.kv_ratio, dim=-2)

        core_attn_out, recurrent_state = self.gated_delta(
            query,
            key,
            value,
            self.A_log,
            self.dt_bias,
            a,
            b,
            recurrent_state=recurrent_state,
            gated_delta_meta=gated_delta_meta,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1
        )

        output = self.out_proj(core_attn_out)
        return output

    Qwen3_5GatedDeltaNet.forward = custom_forward
    Qwen3_5ModelConfigBuilder.build = custom_build
    Qwen3_5ForConditionalGeneration.prepare_inputs_for_generation = (
        custom_prepare_inputs_for_generation
    )


def patch_ray_init():
    """Monkey-patch lmdeploy's init_ray_cluster to register custom NPU resources.

    Ray does not auto-detect Ascend NPUs; without registering custom resources
    at ray.init() time, placement groups requesting ``{'NPU': 1}`` never schedule
    on a fresh local cluster.
    """
    import os
    import logging
    import lmdeploy.pytorch.ray as _ray_mod

    logger = logging.getLogger('dlinfer.ray')
    _orig_init_ray_cluster = _ray_mod.init_ray_cluster

    def _infer_local_ray_custom_resources(device_type, world_size):
        if device_type == 'ascend':
            n = None
            try:
                npu_mod = getattr(torch, 'npu', None)
                if npu_mod is not None and callable(getattr(npu_mod, 'device_count', None)):
                    n = int(npu_mod.device_count())
                    if n <= 0:
                        n = None
            except Exception:
                n = None
            if n is None:
                vis = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '').strip()
                if vis:
                    n = len([x for x in vis.split(',') if x.strip() != ''])
            if n is None or n <= 0:
                n = int(world_size)
                logger.warning(
                    'Could not detect NPU count; registering Ray resource NPU=%d '
                    'from world_size.', n)
            return {'NPU': float(n)}
        if device_type == 'camb':
            n = None
            try:
                mlu = getattr(torch, 'mlu', None)
                if mlu is not None and callable(getattr(mlu, 'device_count', None)):
                    n = int(mlu.device_count())
                    if n <= 0:
                        n = None
            except Exception:
                n = None
            if n is None or n <= 0:
                n = int(world_size)
                logger.warning('Could not detect MLU count; registering MLU=%d.', n)
            return {'MLU': float(n)}
        return None

    def _patched_init_ray_cluster(world_size, ray_address=None, dp=1, device_type='cuda'):
        """Same as original but registers custom resources at ray.init() for local clusters."""
        import ray
        if not ray.is_initialized():
            num_cpus = world_size
            object_store_memory = _ray_mod._get_obj_store_memory(dp=dp)
            init_kwargs = dict(
                ignore_reinit_error=True,
                num_cpus=num_cpus,
                object_store_memory=object_store_memory,
            )
            if ray_address is not None:
                init_kwargs['address'] = ray_address
            if ray_address is None:
                custom_res = _infer_local_ray_custom_resources(device_type, world_size)
                if custom_res:
                    init_kwargs['resources'] = custom_res
            try:
                ray.init(**init_kwargs)
            except ValueError as e:
                if e.args is not None and len(e.args) >= 1 and e.args[
                        0] == 'When connecting to an existing cluster, num_cpus and num_gpus must not be provided.':
                    ray.init(address=ray_address, ignore_reinit_error=True)
                else:
                    raise

        # Remaining logic unchanged from original init_ray_cluster
        device_str = _ray_mod.get_device_str(device_type)
        current_placement_group = ray.util.get_current_placement_group()
        owned_pg = False
        if not current_placement_group:
            num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
            if world_size > num_devices_in_cluster:
                _ray_mod.logger.warning(
                    'The number of required %ss exceeds the total '
                    'number of available %ss in the placement group.', device_str, device_str)
            placement_group_specs = [{device_str: 1.0} for _ in range(world_size)]
            current_ip = ray.util.get_node_ip_address()
            placement_group_specs[0][f'node:{current_ip}'] = 0.001
            current_placement_group = ray.util.placement_group(placement_group_specs, strategy='PACK')
            _ray_mod._wait_until_pg_ready(current_placement_group)
            owned_pg = True

        assert current_placement_group is not None
        placement_group = current_placement_group
        return placement_group, owned_pg

    _ray_mod.init_ray_cluster = _patched_init_ray_cluster


def vendor_device_init():
    import_vendor_module(vendor_name)
    patch_compiled_func()
    patch_async_sampling_logits()
    if vendor_name in ["camb", "ascend"]:
        patch_contiguous_cache_engine()
    if vendor_name == "ascend":
        patch_rejection_sampler()
        patch_spec_decode_runtime()
        patch_state_cache_engine()
        patch_gated_delta_net()      # MUST be before patch_attention_is_tp
        patch_attention_is_tp()
        patch_qwen3_5()
        patch_ray_init()


vendor_device_init()
