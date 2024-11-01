import torch
import torch.fx
from torch.fx import replace_pattern
from torch.fx.node import Argument, Target
import torch.fx.traceback as fx_traceback
from torch.fx.proxy import Proxy
from typing import Any, Dict, Tuple
from dlinfer.graph.dicp.dynamo_bridge.torch_version import is_torch_210_or_higher
from dlinfer.graph.dicp.dynamo_bridge.utils import symint_in_shape


class OpSetTransformer:
    def __init__(self, patterns):
        self._patterns = patterns

    def transform(self, module: torch.fx.GraphModule):
        # first step: replace pattern
        for pat in self._patterns:
            replace_pattern(module, pat.pattern, pat.replacement)
        return module


class SingleOpTransformer(torch.fx.Transformer):
    def __init__(self, module, conversions):
        super().__init__(module)
        self._conversions = conversions
        self.sym_to_inputs = {}
        self.sym_in_args = {}

    def placeholder(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        proxy = super().placeholder(target, args, kwargs)
        proxy.node.meta = fx_traceback.get_current_meta()
        fake_tensor = proxy.node.meta["val"]
        if isinstance(fake_tensor, torch.SymInt):
            self.sym_to_inputs[fake_tensor.node.str()] = proxy
        elif symint_in_shape(fake_tensor.shape):
            # mention symint position in args
            # dynamic shape feature
            for idx, dim in enumerate(fake_tensor.shape):
                if isinstance(dim, torch.SymInt):
                    st = dim.node.str()
                    if st not in self.sym_in_args:
                        self.sym_in_args[st] = (proxy, idx)
        return proxy

    def get_proxy(
        self, target, args: Tuple[Argument, ...], kwargs: Dict[str, Any] = {}
    ):
        proxy = self.tracer.create_proxy(
            "call_function", target.get_singleton(), args, kwargs
        )
        return proxy

    def get_proxy_from_node(self, node):
        return self.tracer.proxy(node)

    def call_function(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if target in self._conversions:
            converted_target = self._conversions[target]
            if isinstance(converted_target, tuple):
                # converted_target: (Operation, process_args_kwargs_fn)
                out, process_fn = converted_target
                args, kwargs = process_fn(args, kwargs)
            else:
                out = self._conversions[target](self, *args, **kwargs)
            if isinstance(out, Proxy):
                out.node.meta = fx_traceback.get_current_meta()
                return out
            try:
                proxy = self.tracer.create_proxy("call_function", out, args, kwargs)
            except Exception as e:
                raise RuntimeError("tracer create_proxy failed!")

            proxy.node.meta = fx_traceback.get_current_meta()
            return proxy
        return super().call_function(target, args, kwargs)

    def get_attr(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        proxy = super().get_attr(target, args, kwargs)
        proxy.node.meta = fx_traceback.get_current_meta()
        if "val" not in proxy.node.meta:
            proxy.node.meta["val"] = self.fetch_attr(target)
        return proxy


if is_torch_210_or_higher:
    import functools
    import inspect
    from typing import List
    from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._inductor.pattern_matcher import (
        PatternMatcherPass,
        Match,
        stable_topological_sort,
        register_replacement,
    )

    def symbolic_trace_ignore_args(fn, args):
        return torch.fx.symbolic_trace(fn)

    class BackendPatternBase:
        trace_fn = symbolic_trace_ignore_args

        @staticmethod
        def pattern(*args, **kwargs):
            raise NotImplementedError("pattern is not implemented")

        @staticmethod
        def replacement(*args, **kwargs):
            raise NotImplementedError("replacement is not implemented")

        @classmethod
        def gen_args(cls):
            return [None] * (cls.pattern.__code__.co_argcount)

        @classmethod
        def gen_tensor(cls, shape=(10, 10), dtype=torch.float16):
            return torch.empty(shape, dtype=dtype, device="cuda")

        @classmethod
        def check_fn(cls, match: Match):
            if match.replacement_graph is None:
                argnames = [*inspect.signature(cls.pattern).parameters.keys()]
                args = list(
                    torch.fx.map_arg(
                        [match.kwargs[name] for name in argnames],
                        lambda n: n.meta["val"],
                    )
                )
                with torch._dynamo.utils.detect_fake_mode(args):
                    match.replacement_graph = cls.trace_fn(
                        cls.replacement, cls.gen_args()
                    )
            return True

        @classmethod
        @functools.lru_cache(None)
        def register(cls, backend_patterns):
            pattern_expr = register_replacement(
                cls.pattern,
                cls.replacement,
                cls.gen_args(),
                cls.trace_fn,
                backend_patterns,
                extra_check=cls.check_fn,
            )
            pattern_entries = backend_patterns[pattern_expr.fns[0]]
            registered_pattern_entry = [
                entry for entry in pattern_entries if entry.pattern == pattern_expr
            ][0]
            registered_pattern_entry.extra_check = cls.check_fn

    def register_backend_patterns(
        patterns_cls_list: List[BackendPatternBase], Pattern: BackendPatternBase
    ):
        patterns_cls_list.append(Pattern)
        return Pattern

    @functools.lru_cache(None)
    def lazy_register_backend_patterns(
        patterns: PatternMatcherPass, patterns_cls_list: Tuple[BackendPatternBase]
    ):
        with torch._guards.tracing(
            None
        ), maybe_disable_fake_tensor_mode(), FakeTensorMode():
            for pattern in patterns_cls_list:
                pattern.register(patterns)

    class BackendPatternMatcherTransformer:
        def __init__(
            self,
            patterns: PatternMatcherPass,
            patterns_cls_list: List[BackendPatternBase],
        ):
            self._patterns = patterns
            lazy_register_backend_patterns(self._patterns, tuple(patterns_cls_list))

        def transform(self, module: torch.fx.GraphModule):
            match_count = self._patterns.apply(module)
            if match_count:
                stable_topological_sort(module.graph)
                module.graph.lint()
                module.recompile()
            return module
