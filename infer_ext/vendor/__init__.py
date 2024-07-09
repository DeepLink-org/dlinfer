import importlib
from pathlib import Path
import torch

def vendor_torch_init():
    vendor_name_file = Path(__file__).parent / "vendor"
    with open(str(vendor_name_file), "r") as f:
        vendor_name = f.read().strip()
    importlib.import_module(f".{vendor_name}", __package__)

vendor_torch_init()
device_type = torch.device(0).type
vendor_ops_registry=dict()
