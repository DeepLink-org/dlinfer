import math
import json
import torch
import torch.distributed as dist
from torch.fx.node import Node
from torch.utils._pytree import tree_map_only
from torch._inductor.utils import IndentedBuffer
from dlinfer.graph.dicp.vendor.AtbGraph.codegen import atb_infer_param as infer_param
from dlinfer.graph.dicp.vendor.AtbGraph.codegen.atb_graph import (
    Operation,
    SqueezeOperation,
    GetItemOperation,
    GraphOpearation,
    UnsqueezeOperation,
    InplaceOperation,
    ViewOperation,
    TupleOperation,
)
from dlinfer.graph.dicp.vendor.AtbGraph.codegen.utils import get_acl_dtype


class AtbOverrides:
    def gen_args(op_var, args_dict, args):
        src_code = IndentedBuffer()
        args_str = [op_var]
        args_str.extend(tree_map_only(Node, lambda x: args_dict[x.name], args))
        return src_code, args_str

    def Linear(name, a, b, bias, trans_a, trans_b, out_dtype=None):
        op = Operation(name, "LinearOperation")
        param = infer_param.LinearParam()
        param.transposeA = trans_a
        param.transposeB = trans_b

        op.set_input([a, b])
        if bias:
            param.hasBias = True
            op.add_input(bias)
        else:
            param.hasBias = False
        if out_dtype:
            assert "now out_dtype cannot set!"
        op.set_param(param)
        op.set_output([name])
        return op

    def LinearAllReduce(name, x, weight, bias):
        op = Operation(name, "LinearParallelOperation")
        param = infer_param.LinearParallelParam()

        param.transWeight = True
        param.rank = dist.get_rank()
        param.rankSize = dist.get_world_size()
        param.rankRoot = 0
        param.hasResidual = False
        param.parallelType = infer_param.ParallelType.LINEAR_ALL_REDUCE
        param.backend = "lccl"

        if bias:
            op.set_input([x, weight, bias])
        else:
            op.set_input([x, weight])
        op.set_param(param)
        op.set_output([name])
        return op

    def AllReduce(name, x, reduce_type):
        op = Operation(name, "AllReduceOperation")
        param = infer_param.AllReduceParam()
        param.rank = dist.get_rank()
        param.rankSize = dist.get_world_size()
        param.rankRoot = 0
        param.allReduceType = reduce_type
        param.backend = "lccl"
        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def Add(name, x, y):
        op = Operation(name, "ElewiseOperation")
        param = infer_param.ElewiseParam()
        param.elewiseType = infer_param.ElewiseType.ELEWISE_ADD

        op.set_input([x, y])
        op.set_param(param)
        op.set_output([name])
        return op

    def Adds(name, x, y, dtype="FLOAT"):
        op = Operation(name, "AclNnAddsOperation")
        param = infer_param.AddsParam()
        param.name = name
        param.value = y
        param.dtype = dtype

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def Sub(name, x, y):
        op = Operation(name, "ElewiseOperation")
        param = infer_param.ElewiseParam()
        param.elewiseType = infer_param.ElewiseType.ELEWISE_SUB

        op.set_input([x, y])
        op.set_param(param)
        op.set_output([name])
        return op

    def Subs(name, x, y, dtype="FLOAT"):
        op = Operation(name, "AclNnSubsOperation")
        param = infer_param.SubsParam()
        param.name = name
        param.value = y
        param.dtype = dtype

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def Div(name, x, y):
        op = Operation(name, "ElewiseOperation")
        param = infer_param.ElewiseParam()
        param.elewiseType = infer_param.ElewiseType.ELEWISE_REALDIV

        op.set_input([x, y])
        op.set_param(param)
        op.set_output([name])
        return op

    def Divs(name, x, y, dtype="FLOAT"):
        op = Operation(name, "AclNnDivsOperation")
        param = infer_param.DivsParam()
        param.name = name
        param.divisor = float(y)
        param.dtype = dtype

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def Mul(name, x, y):
        op = Operation(name, "ElewiseOperation")
        param = infer_param.ElewiseParam()
        param.elewiseType = infer_param.ElewiseType.ELEWISE_MUL

        op.set_input([x, y])
        op.set_param(param)
        op.set_output([name])
        return op

    def Muls(name, x, y, dtype="FLOAT"):
        op = Operation(name, "AclNnMulsOperation")
        param = infer_param.MulsParam()
        param.name = name
        param.value = float(y)
        param.dtype = dtype

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def PowTensorTensor(name, x, y):
        op = Operation(name, "AclNnPowTensorTensorOperation")
        param = infer_param.PowTensorTensorParam()
        param.name = name

        op.set_input([x, y])
        op.set_param(param)
        op.set_output([name])
        return op

    def PowTensorScalar(name, x, y, dtype="FLOAT"):
        op = Operation(name, "AclNnPowTensorScalarOperation")
        param = infer_param.PowTensorScalarParam()
        param.name = name
        param.exponent = float(y)
        param.dtype = dtype

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def Max(name, x):
        op = Operation(name, "AclNnMaxOperation")
        param = infer_param.MaxParam()
        param.name = name

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def Reciprocal(name, x):
        op = Operation(name, "AclNnReciprocalOperation")
        param = infer_param.ReciprocalParam()
        param.name = name

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def Where(name, cond, x, y):
        op = Operation(name, "AclNnSWhereOperation")
        param = infer_param.WhereParam()
        param.name = name

        op.set_input([cond, x, y])
        op.set_param(param)
        op.set_output([name])
        return op

    def Arange(name, start, end, step):
        op = Operation(name, "AclNnArangeOperation")
        param = infer_param.ArangeParam()
        param.name = name
        param.start = start
        param.end = end
        param.step = step

        op.set_param(param)
        op.set_output([name])
        return op

    def GtScalar(name, x, y, dtype="FLOAT"):
        op = Operation(name, "AclNnGtScalarOperation")
        param = infer_param.GtScalarParam()
        param.name = name
        param.value = float(y)
        param.dtype = dtype

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def Graph(name, *args, **kwargs):
        outputs = kwargs["output"]
        if not isinstance(outputs, list):
            outputs = [outputs]

        infer_shape = None
        if "infer_shape" in kwargs.keys():
            infer_shape = kwargs["infer_shape"]

        graph_output_names = []
        for x in outputs:
            if isinstance(x, torch.fx.node.Node) and isinstance(x.meta["val"], list):
                meta_val = x.meta["val"]
                if len(meta_val) != 1:
                    node_name = str(x)
                    for i in range(len(meta_val)):
                        graph_output_names.append(f"{node_name}__{i}")
                    continue
            graph_output_names.append(str(x))

        op = GraphOpearation(name)
        op.set_node_names(list(args))
        op.set_output(graph_output_names)
        if infer_shape:
            op.has_infer_shape = True
            op.infer_shape = infer_shape
        return op

    def GetItem(name, x, index):
        op = GetItemOperation(name)
        op.set_input([x])
        op.index = index
        op.set_output([name])
        return op

    def RmsNorm(name, x, w, eps):
        op = Operation(name, "RmsNormOperation")
        param = infer_param.RmsNormParam()
        param.layerType = infer_param.RmsNormType.RMS_NORM_NORM
        param.normParam.epsilon = eps
        param.normParam.rstd = False

        op.set_input([x, w])
        op.set_param(param)
        op.set_output([name])
        return op

    def Rope(name, query, key, cos, sin, seqlen):
        op = Operation(name, "RopeOperation")
        param = infer_param.RopeParam()
        param.rotaryCoeff = 2

        if seqlen is None:
            # special hack for non-input param seqlen
            seqlen = "rope_seqlen_default"
            op.add_special_constants(
                seqlen, 'torch.ones([1], device="npu", dtype=torch.int32)'
            )
        op.set_input([query, key, cos, sin, seqlen])
        op.set_param(param)
        op.set_output([f"{name}__0", f"{name}__1"])
        return op

    def Inplace(name, input, target, input_index=-1, target_index=-1):
        op = InplaceOperation(name)
        op.input_index = input_index
        op.target_index = target_index
        op.target = target
        op.set_input([input])
        op.set_output([name])
        return op

    def SelfAttentionPAEncoder(
        name, query, key, value, seqlen, mask, q_head_num, kv_head_num
    ):
        op = Operation(name, "SelfAttentionOperation")
        param = infer_param.SelfAttentionParam()
        param.calcType = infer_param.SelfAttentionCalcType.PA_ENCODER
        # param.kernelType = infer_param.SelfAttentionKernelType.KERNELTYPE_DEFAULT
        param.kernelType = infer_param.SelfAttentionKernelType.KERNELTYPE_HIGH_PRECISION
        param.clampType = infer_param.SelfAttentionClampType.CLAMP_TYPE_UNDEFINED
        param.headNum = q_head_num
        param.kvHeadNum = kv_head_num
        param.qkScale = 1.0 / math.sqrt(128)
        param.isTriuMask = 1

        if mask is not None:
            param.maskType = infer_param.SelfAttentionMaskType.MASK_TYPE_NORM
            op.set_input([query, key, value, mask, seqlen])
        else:
            param.maskType = infer_param.SelfAttentionMaskType.MASK_TYPE_UNDEFINED
            op.set_input([query, key, value, seqlen])

        op.set_param(param)
        op.set_output([name])
        op.has_host_inputs = True
        op.host_inputs.append(seqlen)
        return op

    def ReshapeAndCache(name, key, value, key_cache, value_cache, kv_indices):
        op = Operation(name, "ReshapeAndCacheOperation")
        param = infer_param.ReshapeAndCacheParam()

        op.set_param(param)
        op.set_input([key, value, key_cache, value_cache, kv_indices])
        op.set_output([f"{name}__0", f"{name}__1"])
        op.has_inplace_output = True
        op.add_inplace_output(0, 2)
        op.add_inplace_output(1, 3)
        return op

    def PagedAttention(
        name,
        query,
        key_cache,
        value_cache,
        block_table,
        context_len,
        mask,
        q_head_num,
        kv_head_num,
        scale,
    ):
        op = Operation(name, "PagedAttentionOperation")
        param = infer_param.PagedAttentionParam()
        param.headNum = q_head_num
        param.kvHeadNum = kv_head_num
        param.qkScale = scale

        if mask is not None:
            param.maskType = infer_param.PagedAttentionMaskType.MASK_TYPE_NORM
            op.set_input(
                [query, key_cache, value_cache, block_table, context_len, mask]
            )
        else:
            param.maskType = infer_param.PagedAttentionMaskType.UNDEFINED
            op.set_input([query, key_cache, value_cache, block_table, context_len])
        op.set_param(param)
        op.set_output([name])
        op.has_host_inputs = True
        op.host_inputs.append(context_len)
        return op

    def AddRmsNorm(name, x1, x2, gamma, epsilon):
        op = Operation(name, "AddRmsNormOperation")
        param = infer_param.AddRmsNormParam()
        param.epsilon = epsilon
        op.set_param(param)
        op.set_input([x1, x2, gamma])
        op.set_output([f"{name}__0", f"{name}__1", f"{name}__2"])
        return op

    def Transpose(name, x, perm):
        op = Operation(name, "AclNnPermuteOperation")
        # op = Operation(name, "TransposeOperation")
        param = infer_param.TransposeParam(name, perm)

        op.set_param(param)
        op.set_input([x])
        op.set_output([name])
        return op

    def View(name, input, size):
        op = ViewOperation(name)
        op.add_input(input)
        op.add_output(name)
        op.target_shape = size
        op.target_reshape_info = {
            "reshapeType": "view",
            "dimNum": len(size),
            "dims": size,
        }
        return op

    def Tuple(name, *args, **kwargs):
        op = TupleOperation(name)
        op.set_input(list(args))
        op.set_output([name])
        return op

    def SplitSharing(name, x, size, dim):
        op = Operation(name, "SplitOperation")
        param = infer_param.SplitParam()
        param.splitDim = dim
        param.splitNum = len(size)
        op.set_param(param)
        op.set_input([x])
        if len(size) == 2:
            op.set_output([f"{name}__0", f"{name}__1"])
        else:
            op.set_output([f"{name}__0", f"{name}__1", f"{name}__2"])
        return op

    def SplitWithSize(name, x, sizes, dim):
        op = Operation(name, "SplitWithSizeOperation")
        param = infer_param.SplitParam()
        param.splitDim = dim
        param.splitSizes = sizes
        op.set_param(param)
        op.set_input([x])
        for idx, _ in enumerate(sizes):
            op.add_output(f"{name}__{idx}")
        return op

    def Swish(name, x, scale=1.0, dim=-1):
        op = Operation(name, "ActivationOperation")
        param = infer_param.ActivationParam()
        param.activationType = infer_param.ActivationType.ACTIVATION_SWISH.value
        param.scale = scale
        param.dim = dim
        op.set_param(param)
        op.set_input([x])
        op.set_output([name])
        return op

    @staticmethod
    def Cast(name, x, out_dtype):
        # op = Operation(name, "ElewiseOperation")
        # param = infer_param.ElewiseParam()
        # param.elewiseType = infer_param.ElewiseType.ELEWISE_CAST

        op = Operation(name, "AclNnCastOperation")
        param = infer_param.AclNnCastParam()
        param.name = name
        acl_dtype = get_acl_dtype(out_dtype)
        param.outTensorType = acl_dtype
        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    @staticmethod
    def Sin(name, x):
        op = Operation(name, "ElewiseOperation")
        param = infer_param.ElewiseParam()
        param.elewiseType = infer_param.ElewiseType.ELEWISE_SIN

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    @staticmethod
    def Cos(name, x):
        op = Operation(name, "ElewiseOperation")
        param = infer_param.ElewiseParam()
        param.elewiseType = infer_param.ElewiseType.ELEWISE_COS

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    @staticmethod
    def Concat(name, x, dim):
        op = Operation(name, "AclNnCatOperation")
        param = infer_param.AclNnConcatParam()
        param.name = name
        param.concatDim = dim
        param.inputNum = len(x)

        op.set_input(x)
        op.set_param(param)
        op.set_output([name])
        return op

    @staticmethod
    def BatchMatMul(name, x1, x2):
        op = Operation(name, "AclNnBatchMatMulOperation")
        param = infer_param.BatchMatMulParam()
        param.cubeMathType = 1
        param.name = name

        op.set_input([x1, x2])
        op.set_param(param)
        op.set_output([name])
        return op

    def Unsqueeze(name, input, dim):
        op = UnsqueezeOperation(name)
        op.add_input(input)
        op.add_output(name)
        op.dim = [dim]
        op.target_reshape_info = {
            "reshapeType": "unsqueeze",
            "dim": [dim],
        }
        return op

    def Squeeze(name, input, dim):
        op = SqueezeOperation(name)
        op.add_input(input)
        op.add_output(name)
        op.dim = [dim]
        op.target_reshape_info = {
            "reshapeType": "squeeze",
            "dim": [dim],
        }
        return op

    def Gather(name, x1, x2, axis):
        op = Operation(name, "GatherOperation")
        param = infer_param.GatherParam()
        param.axis = axis

        op.set_input([x1, x2])
        op.set_param(param)
        op.set_output([name])
        return op

    def Softmax(name, x, dim):
        op = Operation(name, "AclNnSoftmaxOperation")
        param = infer_param.SoftmaxParam()
        param.name = name
        if not isinstance(dim, list):
            dim = [dim]
        param.axes = dim

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def Sort(name, x, topk):
        op = Operation(name, "AclNnTopkOperation")
        param = infer_param.SortParam()
        param.num = topk

        op.set_input([x])
        op.set_param(param)
        op.set_output([f"{name}__0", f"{name}__1"])
        return op

    def Slice(name, x, dim, offsets, size):
        op = Operation(name, "SliceOperation")
        param = infer_param.SliceParam()
        param.offsets = offsets
        param.size = size

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def AclNnSlice(name, x, dim, start, end, step):
        op = Operation(name, "AclNnSliceOperation")
        param = infer_param.AclNnSliceParam()
        param.name = name
        param.dim = dim
        param.start = start
        param.end = end
        param.step = step

        op.set_input([x])
        op.set_param(param)
        op.set_output([name])
        return op

    def IndexSelect(name, x, dim, index):
        op = Operation(name, "AclNnIndexSelectOperation")
        param = infer_param.IndexSelectParam()
        param.name = name
        param.dim = dim

        op.set_input([x, index])
        op.set_param(param)
        op.set_output([name])
        return op
