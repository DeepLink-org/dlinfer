import dlinfer.pytorch_patch as pytorch_patch
import dlinfer.vendor as vendor

vendor.vendor_torch_init()
vendor.load_extension_ops()
