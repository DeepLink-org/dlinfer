# SPDX-License-Identifier: Apache-2.0
#
# Rejection sampling for speculative decoding on Ascend NPU.
#
# Triton kernels adapted from:
#   vllm-ascend: vllm_ascend/ops/triton/reject_sample.py
#   (Apache-2.0, Copyright (c) 2025 Huawei Technologies Co., Ltd.)
#
# PyTorch fallbacks adapted from:
#   vllm-ascend: vllm_ascend/sample/rejection_sampler.py
#   (Apache-2.0, Copyright (c) 2025 Huawei Technologies Co., Ltd.)
#
# Interface wrapper designed for lmdeploy's rejection_sample signature.
# mypy: ignore-errors

import torch

PLACEHOLDER_TOKEN_ID = -1

# ---------------------------------------------------------------------------
# Try to load Triton and the Ascend CANN get_element extension.
# The `get_element` op is required to extract scalars from Triton vectors
# inside per-request loops on Ascend hardware.
# ---------------------------------------------------------------------------
_TRITON_AVAILABLE = False
_get_element = None

try:
    import triton
    import triton.language as tl

    try:
        import triton.language.extra.cann.extension as _cann_ext
        _get_element = getattr(_cann_ext, "get_element", None)
    except ImportError:
        _get_element = getattr(tl, "get_element", None)

    _TRITON_AVAILABLE = _get_element is not None
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Grid / block-size helper
# ---------------------------------------------------------------------------

def _cal_grid_block_size(batch_size: int):
    """Return (grid, block_size) tuned to the NPU's vector-core count."""
    try:
        from .triton_utils import init_device_properties_triton, get_vectorcore_num
        init_device_properties_triton()
        vectorcore_num = get_vectorcore_num()
    except Exception:
        vectorcore_num = max(1, batch_size)

    if batch_size <= vectorcore_num:
        return batch_size, 1
    grid = vectorcore_num
    block_size = triton.next_power_of_2(triton.cdiv(batch_size, grid))
    return grid, block_size


# ---------------------------------------------------------------------------
# Triton kernels  (compiled only when _TRITON_AVAILABLE is True)
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    get_element = _get_element

    @triton.jit(do_not_specialize=["max_spec_len"])
    def _bonus_store_kernel(
        bonus_token_ids_ptr,
        position,
        output_token_ids_ptr,
        max_spec_len,
        num_accepted,
    ):
        bonus = tl.load(bonus_token_ids_ptr + position)
        tl.store(output_token_ids_ptr + position * (max_spec_len + 1) + num_accepted, bonus)

    @triton.jit(do_not_specialize=["vec_len", "max_spec_len"])
    def _rejection_greedy_kernel(
        output_token_ids_ptr,     # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens_ptr,  # [batch_size]
        draft_token_ids_ptr,      # [num_tokens]
        target_argmax_ptr,        # [num_tokens]
        bonus_token_ids_ptr,      # [batch_size]
        is_greedy_ptr,            # [batch_size] bool, or None (means all greedy)
        vec_len,
        max_spec_len,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_idx = tl.program_id(0)
        offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < vec_len

        if is_greedy_ptr is None:
            is_greedy_mask = mask
        else:
            is_greedy = tl.load(is_greedy_ptr + offset, mask=mask, other=0)
            is_greedy_mask = mask & (is_greedy != 0)

        start_idx = tl.where(
            offset == 0, 0,
            tl.load(cu_num_draft_tokens_ptr + offset - 1, is_greedy_mask),
        )
        end_idx = tl.load(cu_num_draft_tokens_ptr + offset, is_greedy_mask)
        num_draft_tokens = end_idx - start_idx

        for pos in tl.range(0, BLOCK_SIZE):
            n = get_element(num_draft_tokens, (pos,))
            s = get_element(start_idx, (pos,))
            is_g = get_element(is_greedy_mask, (pos,))
            req_idx = block_idx * BLOCK_SIZE + pos
            rejected = False
            for i in range(n):
                if not rejected:
                    draft_id = tl.load(draft_token_ids_ptr + s + i)
                    target_id = tl.load(target_argmax_ptr + s + i)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + i, target_id)
                    if draft_id != target_id:
                        rejected = True
            if not rejected and is_g:
                _bonus_store_kernel(
                    bonus_token_ids_ptr, req_idx,
                    output_token_ids_ptr, max_spec_len, n,
                )

    @triton.jit(do_not_specialize=["max_spec_len"])
    def _rejection_random_kernel(
        output_token_ids_ptr,     # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens_ptr,  # [batch_size]
        draft_token_ids_ptr,      # [num_tokens]
        draft_probs_ptr,          # [num_tokens, vocab_size] or None
        target_probs_ptr,         # [num_tokens, vocab_size]
        bonus_token_ids_ptr,      # [batch_size]
        recovered_token_ids_ptr,  # [num_tokens]
        uniform_probs_ptr,        # [num_tokens]
        is_greedy_ptr,            # [batch_size] bool
        max_spec_len,
        vocab_size,
        vec_len,
        NO_DRAFT_PROBS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_idx = tl.program_id(0)
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vec_len

        is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
        not_greedy_mask = is_greedy == 0

        start_idxs = tl.where(
            offsets == 0, 0,
            tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask),
        )
        end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
        n_draft_tokens = end_idxs - start_idxs

        for req_i in range(BLOCK_SIZE):
            not_greedy = get_element(not_greedy_mask, (req_i,))
            if not_greedy:
                rejected = False
                s = get_element(start_idxs, (req_i,))
                req_idx = block_idx * BLOCK_SIZE + req_i
                n = get_element(n_draft_tokens, (req_i,))
                for pos in range(n):
                    if not rejected:
                        draft_id = tl.load(draft_token_ids_ptr + s + pos)
                        if NO_DRAFT_PROBS:
                            draft_prob = 1.0
                        else:
                            draft_prob = tl.load(draft_probs_ptr + (s + pos) * vocab_size + draft_id)
                        target_prob = tl.load(target_probs_ptr + (s + pos) * vocab_size + draft_id)
                        uniform_prob = tl.load(uniform_probs_ptr + s + pos)
                        if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                            token_id = draft_id
                        else:
                            rejected = True
                            token_id = tl.load(recovered_token_ids_ptr + s + pos)
                        tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)
                if not rejected:
                    bonus = tl.load(bonus_token_ids_ptr + req_idx)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + n, bonus)

    @triton.jit
    def _sample_recovered_tokens_kernel(
        output_token_ids_ptr,     # [num_tokens]
        cu_num_draft_tokens_ptr,  # [batch_size]
        draft_token_ids_ptr,      # [num_tokens]
        draft_probs_ptr,          # [num_tokens, vocab_size] or None
        target_probs_ptr,         # [num_tokens, vocab_size]  read-only
        q_ptr,                    # [batch_size, vocab_size]  exponential samples
        vocab_size,
        PADDED_VOCAB_SIZE: tl.constexpr,
        NO_DRAFT_PROBS: tl.constexpr,
        SUB_BLOCK: tl.constexpr,
    ):
        req_idx = tl.program_id(0)
        start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
        num_draft_tokens = end_idx - start_idx

        pos = tl.program_id(1)
        if pos >= num_draft_tokens:
            return

        loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
        global_recovered_id = -1
        global_max_score = -1.0

        if NO_DRAFT_PROBS:
            # Load draft token id once; mask it out per sub-block via tl.where
            # to avoid in-place modification of target_probs (which could cause
            # memory pipeline hazards on Ascend with multibuffer enabled).
            draft_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            for loop_i in range(loop):
                vocab_start = loop_i * SUB_BLOCK
                vocab_off = vocab_start + tl.arange(0, SUB_BLOCK)
                prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + vocab_off,
                    mask=vocab_off < vocab_size, other=0.0,
                )
                # Zero out the draft-token position without modifying the buffer.
                prob = tl.where(vocab_off == draft_id, 0.0, prob)
                q = tl.load(
                    q_ptr + req_idx * vocab_size + vocab_off,
                    mask=vocab_off < vocab_size, other=float("-inf"),
                )
                score = prob / q
                local_id = tl.argmax(score, axis=-1)
                local_max = get_element(score, (local_id,))
                if local_max > global_max_score:
                    global_max_score = local_max
                    global_recovered_id = vocab_start + local_id
        else:
            for loop_i in range(loop):
                vocab_start = loop_i * SUB_BLOCK
                vocab_off = vocab_start + tl.arange(0, SUB_BLOCK)
                draft_p = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_off,
                    mask=vocab_off < vocab_size, other=0.0,
                )
                target_p = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + vocab_off,
                    mask=vocab_off < vocab_size, other=0.0,
                )
                prob = tl.maximum(target_p - draft_p, 0.0)
                q = tl.load(
                    q_ptr + req_idx * vocab_size + vocab_off,
                    mask=vocab_off < vocab_size, other=float("-inf"),
                )
                score = prob / q
                local_id = tl.argmax(score, axis=-1)
                local_max = get_element(score, (local_id,))
                if local_max > global_max_score:
                    global_max_score = local_max
                    global_recovered_id = vocab_start + local_id

        tl.store(output_token_ids_ptr + start_idx + pos, global_recovered_id)


# ---------------------------------------------------------------------------
# PyTorch fallbacks  (fully vectorised; no Python loops over tokens)
# Adapted from vllm_ascend/sample/rejection_sampler.py
# ---------------------------------------------------------------------------

def _pytorch_rejection_greedy_sample(
    output_token_ids,    # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens, # [batch_size]
    draft_token_ids,     # [num_tokens]
    target_argmax,       # [num_tokens]
    bonus_token_ids,     # [batch_size]
    num_draft_tokens,    # list[int]
    max_spec_len,
    is_greedy=None,      # [batch_size] bool or None (all greedy)
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device

    draft_tokens_per_req = torch.tensor(num_draft_tokens, device=device)
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(num_tokens, device=device) - start_indices[token_req_ids]

    mismatch_global = draft_token_ids != target_argmax

    if max_spec_len == 0:
        first_mismatch_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        pos_matrix = torch.full((batch_size, max_spec_len), -1, dtype=torch.long, device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.full((batch_size, max_spec_len), False, dtype=torch.bool, device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix, max_spec_len * 2)
        first_mismatch_pos, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch = first_mismatch_pos == max_spec_len * 2
        first_mismatch_pos[no_mismatch] = draft_tokens_per_req[no_mismatch]

    copy_len = torch.minimum(first_mismatch_pos + 1, draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1, device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    final_copy_mask = copy_mask & is_greedy.unsqueeze(1)
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[final_copy_mask] = target_argmax[global_idx[final_copy_mask]].to(output_token_ids.dtype)

    needs_bonus = is_greedy & (first_mismatch_pos >= draft_tokens_per_req)
    if needs_bonus.any():
        rows = needs_bonus.nonzero(as_tuple=True)[0]
        cols = draft_tokens_per_req[rows]
        output_token_ids[rows, cols] = bonus_token_ids.squeeze(1)[rows]


def _pytorch_sample_recovered_tokens(
    cu_num_draft_tokens, # [batch_size]
    draft_token_ids,     # [num_tokens]
    draft_probs,         # [num_tokens, vocab_size] or None
    target_probs,        # [num_tokens, vocab_size]
    q,                   # [batch_size, vocab_size] exponential samples
    device,
):
    num_tokens = draft_token_ids.shape[0]
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    batch_size = q.shape[0]
    # Map each flat token index to its batch index.
    cu_start = torch.cat([
        torch.tensor([0], device=device),
        cu_num_draft_tokens[:-1],
    ])
    token_idx = torch.arange(num_tokens, device=device)
    # in_range[t, b] == True iff token t belongs to request b
    in_range = (token_idx[:, None] >= cu_start[None, :]) & (token_idx[:, None] < cu_num_draft_tokens[None, :])
    token_to_batch = torch.argmax(in_range.int(), dim=1)
    has_match = in_range.any(dim=1)
    token_to_batch = torch.where(has_match, token_to_batch, 0)

    if draft_probs is None:
        # N-gram: zero out draft-token probability in target.
        modified = target_probs.clone()
        modified[token_idx, draft_token_ids] = 0.0
        prob = modified
    else:
        prob = torch.maximum(
            target_probs - draft_probs,
            torch.tensor(0.0, device=device),
        )

    q_per_token = q[token_to_batch]           # [num_tokens, vocab_size]
    eps = 1e-10
    q_safe = q_per_token.clamp(min=eps)
    score = prob / q_safe
    # Suppress padded / zero-q positions.
    score = torch.where((q_per_token == 0) | torch.isinf(q_per_token), -1e10, score)

    return torch.argmax(score, dim=1)


def _pytorch_rejection_random_sample(
    output_token_ids,    # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens, # [batch_size]
    draft_token_ids,     # [num_tokens]
    draft_probs,         # [num_tokens, vocab_size] or None
    target_probs,        # [num_tokens, vocab_size]
    bonus_token_ids,     # [batch_size, 1]
    recovered_token_ids, # [num_tokens]
    uniform_probs,       # [num_tokens]
    is_greedy,           # [batch_size] bool
    max_spec_len,
):
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    cu_start = torch.cat([torch.tensor([0], device=device), cu_num_draft_tokens[:-1]])
    num_draft_per_batch = cu_num_draft_tokens - cu_start  # [batch_size]

    pos_indices = torch.arange(max_spec_len, device=device)[None, :]  # [1, max_spec_len]
    valid_mask = pos_indices < num_draft_per_batch[:, None]
    global_idx = (cu_start[:, None] + pos_indices).clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_idx]         # [batch, max_spec_len]

    if draft_probs is None:
        draft_token_probs = torch.ones(batch_size, max_spec_len, device=device, dtype=torch.float32)
    else:
        flat_idx = global_idx.flatten()
        flat_draft = draft_tokens.flatten()
        draft_token_probs = draft_probs[flat_idx, flat_draft].view(batch_size, max_spec_len)

    flat_idx = global_idx.flatten()
    flat_draft = draft_tokens.flatten()
    target_token_probs = target_probs[flat_idx, flat_draft].view(batch_size, max_spec_len)
    uniform_token_probs = uniform_probs[global_idx]
    recovered_tokens = recovered_token_ids[global_idx]

    zero = torch.tensor(0.0, device=device)
    accepted = (draft_token_probs > zero) & (target_token_probs / draft_token_probs >= uniform_token_probs)

    first_rejection = (~accepted) & valid_mask
    default_pos = torch.full((batch_size, 1), max_spec_len, device=device)
    first_reject_pos = torch.where(
        first_rejection.any(dim=1, keepdim=True),
        first_rejection.float().argmax(dim=1, keepdim=True),
        default_pos,
    )

    should_skip = (pos_indices >= first_reject_pos) & valid_mask
    non_greedy = ~is_greedy
    update_mask = non_greedy[:, None] & valid_mask & ~should_skip
    first_reject_mask = (pos_indices == first_reject_pos) & valid_mask & non_greedy[:, None]
    full_update_mask = update_mask | first_reject_mask

    final_tokens = torch.where(
        first_reject_mask,
        recovered_tokens,
        torch.where(accepted & ~should_skip, draft_tokens, output_token_ids[:, :max_spec_len]),
    )
    output_token_ids[:, :max_spec_len] = torch.where(
        full_update_mask, final_tokens, output_token_ids[:, :max_spec_len]
    )

    # Append bonus token when all draft tokens are accepted.
    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    add_bonus = non_greedy & no_rejection
    bonus_col = num_draft_per_batch           # [batch_size]
    bonus_ok = bonus_col <= max_spec_len
    add_bonus = add_bonus & bonus_ok

    seq_len = output_token_ids.shape[1]
    all_pos = torch.arange(seq_len, device=device)[None, :]
    bonus_pos_mask = (all_pos == bonus_col[:, None]) & add_bonus[:, None]
    bonus_expanded = bonus_token_ids.view(-1, 1).expand(-1, seq_len)
    output_token_ids[:] = torch.where(bonus_pos_mask, bonus_expanded, output_token_ids)


# ---------------------------------------------------------------------------
# Internal driver  (mirrors vllm-ascend rejection_sample logic)
# ---------------------------------------------------------------------------

def _rejection_sample_impl(
    draft_token_ids,       # [num_tokens]
    num_draft_tokens,      # list[int], all equal in lmdeploy
    max_spec_len,          # int
    cu_num_draft_tokens,   # [batch_size]
    draft_probs,           # [num_tokens, vocab_size] or None
    target_probs,          # [num_tokens, vocab_size]  (float32)
    bonus_token_ids,       # [batch_size, 1]
    is_greedy,             # [batch_size] bool or None
    all_greedy,            # bool
    all_random,            # bool
):
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    if _TRITON_AVAILABLE:
        grid, block_size = _cal_grid_block_size(batch_size)
    else:
        grid, block_size = None, None

    # ------------------------------------------------------------------
    # Greedy rejection pass
    # ------------------------------------------------------------------
    if not all_random:
        target_argmax = target_probs.argmax(dim=-1)
        if _TRITON_AVAILABLE:
            _rejection_greedy_kernel[(grid,)](
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids.squeeze(1),
                is_greedy,          # None means all-greedy
                batch_size,
                max_spec_len,
                BLOCK_SIZE=block_size,
            )
        else:
            _pytorch_rejection_greedy_sample(
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                num_draft_tokens,
                max_spec_len,
                is_greedy,
            )

        if all_greedy:
            return output_token_ids

    # ------------------------------------------------------------------
    # Random rejection pass — sample recovered tokens, then accept/reject
    # ------------------------------------------------------------------
    uniform_probs = torch.rand(num_tokens, dtype=torch.float64, device=device).to(torch.float32)

    # q ~ Exp(1); recovered token = argmax((target - draft) / q)
    q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    q.exponential_()

    if _TRITON_AVAILABLE:
        recovered_token_ids = torch.empty(num_tokens, dtype=torch.int32, device=device)
        padded_vocab = triton.next_power_of_2(vocab_size)
        _sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            padded_vocab,
            NO_DRAFT_PROBS=draft_probs is None,
            SUB_BLOCK=4096,
            multibuffer=False,
        )
    else:
        recovered_token_ids = _pytorch_sample_recovered_tokens(
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            device,
        )

    if _TRITON_AVAILABLE:
        _rejection_random_kernel[(grid,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids.squeeze(1),
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            batch_size,
            NO_DRAFT_PROBS=draft_probs is None,
            BLOCK_SIZE=block_size,
        )
    else:
        _pytorch_rejection_random_sample(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
        )

    return output_token_ids


# ---------------------------------------------------------------------------
# Public entry point — lmdeploy rejection_sample interface
# ---------------------------------------------------------------------------

def _extract_outputs(output_token_ids, num_spec_tokens):
    """Compute num_rejected_tokens and last_token_ids from output_token_ids.

    output_token_ids: [batch_size, num_spec_tokens + 1]  (int32, PLACEHOLDER=-1)
    Returns: (output_token_ids, num_rejected_tokens, last_token_ids)
    """
    batch_size = output_token_ids.size(0)
    valid_mask = output_token_ids != PLACEHOLDER_TOKEN_ID
    num_accepted = valid_mask.sum(dim=1)
    num_rejected = num_spec_tokens + 1 - num_accepted
    last_ids = output_token_ids[
        torch.arange(batch_size, device=output_token_ids.device),
        num_accepted - 1,
    ]
    return output_token_ids, num_rejected, last_ids


def rejection_sample(
    target_logits,      # [batch_size, num_spec_tokens, vocab_size]
    draft_token_ids,    # [batch_size, num_spec_tokens]
    bonus_token_ids,    # [batch_size]
    sampling_inputs,    # lmdeploy SamplingInputs
    draft_probs=None,   # [batch_size, num_spec_tokens, vocab_size] or None
):
    """Rejection sampler for lmdeploy speculative decoding on Ascend NPU.

    Replaces the NVIDIA-Triton-based rejection_sample with kernels and
    vectorised PyTorch ops that run correctly on Ascend.

    Returns: (output_token_ids, num_rejected_tokens, last_token_ids)
    """
    batch_size, num_spec_tokens, vocab_size = target_logits.shape
    device = target_logits.device

    # ---- Flatten [batch, spec, vocab] → [num_tokens, vocab] ---------------
    num_tokens = batch_size * num_spec_tokens
    target_logits_flat = target_logits.reshape(num_tokens, vocab_size)
    draft_token_ids_flat = draft_token_ids.reshape(num_tokens).to(torch.int32)
    draft_probs_flat = draft_probs.reshape(num_tokens, vocab_size) if draft_probs is not None else None

    # Build cumulative draft-token offsets (uniform: each request has num_spec_tokens).
    cu_num_draft_tokens = (
        torch.arange(1, batch_size + 1, device=device, dtype=torch.int32) * num_spec_tokens
    )
    num_draft_tokens_list = [num_spec_tokens] * batch_size

    # ---- Determine greedy/random policy ------------------------------------
    all_greedy = sampling_inputs.max_top_k == 1
    if all_greedy:
        is_greedy = None          # signal: all requests are greedy
        all_random = False
    elif sampling_inputs.top_k is not None:
        is_greedy = (sampling_inputs.top_k == 1).to(torch.bool).to(device)
        all_random = not is_greedy.any().item()
    else:
        is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)
        all_random = True

    # ---- Convert logits → probs for the non-greedy path -------------------
    target_probs = target_logits_flat.softmax(dim=-1, dtype=torch.float32)

    # ---- Run rejection sampling -------------------------------------------
    output_token_ids = _rejection_sample_impl(
        draft_token_ids=draft_token_ids_flat,
        num_draft_tokens=num_draft_tokens_list,
        max_spec_len=num_spec_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_probs=draft_probs_flat,
        target_probs=target_probs,
        bonus_token_ids=bonus_token_ids.unsqueeze(1),
        is_greedy=is_greedy,
        all_greedy=all_greedy,
        all_random=all_random,
    )

    return _extract_outputs(output_token_ids.long(), num_spec_tokens)
