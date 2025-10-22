# Copyright (c) 2024, DeepLink. All rights reserved.
import torch
import torch_npu
from torch_npu.npu.graphs import (
    _GraphDispatchMode,
    graph_task_group_begin,
    graph_task_group_end,
)


def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if func.__name__ in [
        "npu_fused_infer_attention_score",
        "npu_fused_infer_attention_score.default",
    ]:
        func_out = torch_npu.npu_fused_infer_attention_score.out
        self.update_schema(str(func_out.__name__), str(func_out._schema))
        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        # apply tensor
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            *args, **kwargs
        )
        output = torch.empty_like(args[0])
        softmax_lse = torch.empty(1, dtype=args[0].dtype, device=args[0].device)
        kwargs["workspace"] = workspace
        kwargs["out"] = [output, softmax_lse]
        # begin graph task
        graph_task_group_begin(stream)
        func_out(*args, **kwargs)
        handle = graph_task_group_end(stream)
        # save state for update
        self.graph_dispatch_records.append(
            self._append_dispatch_record(event, handle, args, kwargs, func_out)
        )
        return kwargs["out"]
    elif func.__name__ == "npu_fused_infer_attention_score.out":
        self.update_schema(str(func.__name__), str(func._schema))
        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        # begin graph task
        graph_task_group_begin(stream)
        func(*args, **kwargs)
        handle = graph_task_group_end(stream)
        # save state for update
        self.graph_dispatch_records.append(
            self._append_dispatch_record(event, handle, args, kwargs, func)
        )
        return kwargs["out"]
    else:
        return func(*args, **kwargs)


_GraphDispatchMode.__torch_dispatch__ = __torch_dispatch__
