# Copyright (c) 2024, DeepLink. All rights reserved.
import inspect
import types
from functools import wraps

from torch.library import Library, impl

import dlinfer.graph
from dlinfer.utils.type_annotation import Callable, Optional, Sequence, Dict
from dlinfer.vendor import dispatch_key, vendor_name

library_impl_dict: Dict[str, Library] = dict()
graph_enabled_backends = ["ascend"]


def register_custom_op(
    qualname: str,
    shape_param_keys: Optional[Sequence[str]] = None,
    default_value: Optional[Dict] = None,
    impl_abstract_func: Optional[Callable] = None,
) -> Callable:
    disable = vendor_name not in graph_enabled_backends

    def inner_func(func: Callable):
        if disable:
            return override_default_value_static(default_value)(func)
        import torch._custom_ops

        nonlocal impl_abstract_func
        lib_name, func_name = qualname.split("::")
        torch._custom_ops.custom_op(qualname)(func)
        # using low level torch.library APIs in case of the registration
        # of fallback kernels which raises error in torch._custom_ops.impl
        if lib_name not in library_impl_dict:
            library_impl_dict[lib_name] = Library(lib_name, "IMPL")
        impl(library_impl_dict[lib_name], func_name, dispatch_key)(func)
        if impl_abstract_func is None:
            assert shape_param_keys is not None
            params_name_list = [name for name in inspect.signature(func).parameters]

            def _impl_abstract_func(*args, **kwargs):
                assert len(args) + len(kwargs) == len(params_name_list)
                result = []
                for key in shape_param_keys:
                    key_index = params_name_list.index(key)
                    if key_index < len(args):
                        target = args[key_index]
                    else:
                        target = kwargs[key]
                    result.append(torch.empty_like(target))
                if len(result) == 1:
                    return result[0]
                return tuple(result)

            impl_abstract_func = _impl_abstract_func
        torch._custom_ops.impl_abstract(qualname)(impl_abstract_func)
        torch_ops_namespace = getattr(torch.ops, lib_name)
        torch_ops_func = getattr(torch_ops_namespace, func_name)
        assert torch_ops_func is not None
        # override default value
        func_with_default = override_default_value_static(default_value)(func)
        torch_ops_func_with_default = override_default_value_dynamic(
            default_value, func
        )(torch_ops_func)

        # use config.enable_graph_mode to control func call
        @wraps(func)
        def patched_func(*args, **kwargs):
            if not dlinfer.graph.config.enable_graph_mode:
                return func_with_default(*args, **kwargs)
            else:
                return torch_ops_func_with_default(*args, **kwargs)

        return patched_func

    return inner_func


def override_default_value_dynamic(
    default_value: Optional[Dict], origin_func: Callable
):
    def inner_func(func):
        if default_value is None:
            return func
        sig = inspect.signature(origin_func)
        sig_param_keys = sig.parameters.keys()
        params_str = ", ".join(sig_param_keys)
        params_with_default = []
        for name in sig_param_keys:
            if name in default_value:
                if isinstance(default_value[name], str):
                    params_with_default.append(f"{name}='{default_value[name]}'")
                else:
                    params_with_default.append(f"{name}={default_value[name]}")
            else:
                params_with_default.append(name)
        params_str_with_default = ", ".join(params_with_default)
        func_code = f"""
def {func.__name__}({params_str_with_default}):
    return original_func({params_str})
"""
        exec_namespace = {}
        # it's hard not to use exec here
        exec(func_code, {"original_func": func}, exec_namespace)
        dynamic_func = exec_namespace[func.__name__]

        return dynamic_func

    return inner_func


def override_default_value_static(default_value: Optional[Dict]):
    # suitable for the function which signature isn't (*args, **kwargs)
    def inner_func(func):
        if default_value is None:
            return func
        sig = inspect.signature(func)
        old_params = sig.parameters
        new_params = []
        default_arg = []
        default_kwarg = []
        func_co_argcount = func.__code__.co_argcount
        param_has_default_value = False
        for idx, (name, param) in enumerate(old_params.items()):
            if name in default_value:
                new_param = param.replace(default=default_value[name])
            else:
                new_param = param
            new_params.append(new_param)
            if new_param.default is not inspect._empty:
                if not param_has_default_value:
                    param_has_default_value = True
                if idx < func_co_argcount:
                    default_arg.append(new_param.default)
                else:
                    default_kwarg.append((name, new_param.default))
            else:
                if param_has_default_value:
                    raise SyntaxError(
                        f"non-default argument '{name}' follows default argument"
                    )
        new_signature = sig.replace(parameters=new_params)
        func.__signature__ = new_signature
        func.__defaults__ = tuple(default_arg)
        func.__kwdefaults__ = dict(default_kwarg)
        return func

    return inner_func
