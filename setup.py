import os
import importlib
from pathlib import Path
from setuptools import find_packages
import yaml
import torch
from skbuild import setup


VERSION = "0.0.1"

vendor_dispatch_key_map = {
    "ascend": "PrivateUse1",
    "camb": "PrivateUse1",
}

vendor_torch_map = {
    "ascend": "torch_npu",
    "camb": "torch_mlu",
}

def gen_vendor_yaml(device):
    config = dict()
    config['vendor'] = device
    assert device in vendor_dispatch_key_map
    config['dispatch_key'] = vendor_dispatch_key_map[device]
    file_path = Path(__file__).parent / "infer_ext" / "vendor" / "vendor.yaml"
    with open(str(file_path), "w") as f:
        yaml.safe_dump(config, f)
    return str(file_path.name)

def get_vendor_torch_root(device):
    assert device in vendor_torch_map
    vendor_torch_str = vendor_torch_map[device]
    vendor_torch = importlib.import_module(vendor_torch_str)
    vendor_torch_root = str(Path(vendor_torch.__file__).parent)
    return vendor_torch_str, vendor_torch_root

def get_torch_cxx11_abi():
    return "1" if torch.compiled_with_cxx11_abi() else "0"

def get_torch_cmake_prefix_path():
    return torch.utils.cmake_prefix_path

def get_device():
    return os.getenv("DEVICE", "").lower()

def get_cmake_args():
    cmake_args = list()
    cmake_device = get_device()
    cmake_args.append("-DCMAKE_BUILD_TYPE=Release")
    cmake_args.append(f"-DTorch_DIR={get_torch_cmake_prefix_path()}/Torch")
    cmake_args.append(f"-D_GLIBCXX_USE_CXX11_ABI={get_torch_cxx11_abi()}")
    cmake_args.append(f"-DDEVICE={cmake_device}")
    vendor_torch_str, vendor_torch_root = get_vendor_torch_root(cmake_device)
    cmake_args.append(f"-D{vendor_torch_str.capitalize()}_ROOT={vendor_torch_root}")
    return cmake_args

def get_package_data():
    cmake_device = get_device()
    yaml_file_name = gen_vendor_yaml(cmake_device)
    assert cmake_device, "DEVICE shouldn't be empty!"
    return {
        f"infer_ext.vendor": [
            yaml_file_name,
        ]
    }

def main():
    setup(
        name="infer_ext",
        version=VERSION,
        description="DeepLink Inference Extension",
        url="https://github.com/DeepLink-org/InferExt",
        packages=find_packages(),
        package_data=get_package_data(),
        exclude_package_data={"": ["tests/*"]},
        cmake_args=get_cmake_args(),
        cmake_install_target="all",
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: POSIX :: Linux"
        ],
        python_requires=">=3.8",
        install_requires=[
            "torch >= 2.0.0",
        ]
    )


if __name__ == '__main__':
    main()
