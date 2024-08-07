import infer_ext.pytorch_patch as pytorch_patch
import infer_ext.vendor as vendor
import infer_ext.framework_patch

vendor.vendor_torch_init()
vendor.load_extension_ops()
