import infer_ext.pytorch_patch as pytorch_patch
import infer_ext.vendor as vendor

vendor.vendor_torch_init()
vendor.load_extension_ops()
# vendor.apply_vendor_pytorch_patch()
# pytorch_patch.apply_tensor_method_patch()
# pytorch_patch.apply_torch_function_patch()
# pytorch_patch.apply_dist_patch()
