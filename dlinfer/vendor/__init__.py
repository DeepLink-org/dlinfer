import importlib
from pathlib import Path
from functools import lru_cache
import yaml
import torch


vendor_ops_registry = dict()
vendor_is_initialized = False
apply_vendor_pytorch_patch = None
vendor_name_file = Path(__file__).parent / "vendor.yaml"
with open(str(vendor_name_file), "r") as f:
    config = yaml.safe_load(f)
    vendor_name = config["vendor"]
    dispatch_key = config["dispatch_key"]

@lru_cache(1)
def import_vendor_module(vendor_name_str):
    return importlib.import_module(f".{vendor_name_str}", __package__)

def vendor_torch_init():
    vendor_module = import_vendor_module(vendor_name)
    global device_type, vendor_is_initialized, apply_vendor_pytorch_patch
    vendor_is_initialized = True
    device_type = torch.device(0).type
    apply_vendor_pytorch_patch = vendor_module.apply_vendor_pytorch_patch

@lru_cache(1)
def get_device_str():
    if not vendor_is_initialized:
        vendor_torch_init()
    vendor_module = import_vendor_module(vendor_name)
    return vendor_module.device_str

@lru_cache(1)
def get_comm_str():
    if not vendor_is_initialized:
        vendor_torch_init()
    vendor_module = import_vendor_module(vendor_name)
    return vendor_module.comm_str

def load_extension_ops():
    if not vendor_is_initialized:
        vendor_torch_init()
    extension_file_path = f"{str(Path(__file__).parent / vendor_name / vendor_name)}_extension.so"
    if Path(extension_file_path).exists():
        torch.ops.load_library(extension_file_path)
