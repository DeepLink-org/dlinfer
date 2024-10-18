# Copyright (c) 2024, DeepLink. All rights reserved.
import inspect
from functools import wraps

import torch._custom_ops
from torch.library import Library, impl

from dlinfer.utils.type_annotation import Callable, Optional, Sequence, Dict
from dlinfer.vendor import dispatch_key

library_impl_dict: Dict[str, Library] = dict()


def register_custom_op(
    qualname: str,
    shape_param_keys: Optional[Sequence[str]] = None,
    impl_abstract_func: Optional[Callable] = None,
    disable=False,  # disable graph custom op registration for now
) -> Callable:
    def inner_func(func: Callable):
        if disable:
            return func
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
        wrapped_func = wraps(func)(torch_ops_func)
        return wrapped_func

    return inner_func


def register_custom_op_default_value(kwargs_default_dict: Dict):
    def update_func_kwargs_value(func):
        @wraps(func)
        def inner_func(*args, **kwargs):
            kwargs_default_dict.update(kwargs)
            return func(*args, **kwargs_default_dict)

        return inner_func

    return update_func_kwargs_value
