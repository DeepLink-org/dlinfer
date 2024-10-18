import functools
import torch
from dlinfer.graph.dicp.dynamo_bridge.operator import Operator


def args_kwargs_unchange(args, kwargs):
    return args, kwargs


def register_conversion_impl(
    conversions: list, aten_fn, decomp_fn, process_args_kwargs_fn=None
):
    register_op_singleton_flag = isinstance(decomp_fn, type) and issubclass(
        decomp_fn, Operator
    )
    if register_op_singleton_flag:
        wrapped = (
            decomp_fn.get_singleton(),
            (
                args_kwargs_unchange
                if process_args_kwargs_fn is None
                else process_args_kwargs_fn
            ),
        )
    else:

        @functools.wraps(decomp_fn)
        def wrapped(*args, **kwargs):
            return decomp_fn(*args, **kwargs)

    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    aten_fn_for_key = []
    for fn in list(aten_fn):
        if isinstance(fn, str):
            assert fn.startswith("torch.ops")
            real_fn_name = fn.replace("torch.ops.", "")
            ns, op_overload = real_fn_name.split(".", 1)
            if not hasattr(torch.ops, ns):
                print(
                    f"[dicp] can't find torch.ops.{ns}, conversion for {fn} is ignored"
                )
                continue
            ns_obj = getattr(torch.ops, ns)
            if "." in op_overload:
                op, overload = op_overload.split(".", 1)
                if not hasattr(ns_obj, op):
                    print(
                        f"[dicp] can't find torch.ops.{ns}.{op}, conversion for {fn} is ignored"
                    )
                    continue
                op_obj = getattr(ns_obj, op)
                fn = getattr(op_obj, overload)
            else:
                if not hasattr(ns_obj, op_overload):
                    print(
                        f"[dicp] can't find torch.ops.{ns}.{op_overload}, conversion for {fn} is ignored"
                    )
                    continue
                fn = getattr(ns_obj, op_overload)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                if other_fn not in conversions:
                    aten_fn_for_key.append(other_fn)
        aten_fn_for_key.append(fn)

    conversions.update({fn: wrapped for fn in aten_fn_for_key})
    if register_op_singleton_flag:
        return wrapped[0]
    else:
        return wrapped
