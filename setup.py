import os
import importlib
from pathlib import Path
from setuptools import find_packages
import torch
from skbuild import setup


VERSION = "0.0.1"

vendor_torch_map = {
    "ascend": "torch_npu"
}

def get_vendor_torch_root(device):
    device_lower = device.lower()
    assert device_lower in vendor_torch_map
    vendor_torch_str = vendor_torch_map[device_lower]
    vendor_torch = importlib.import_module(vendor_torch_str)
    vendor_torch_root = str(Path(vendor_torch.__file__).parent)
    return vendor_torch_str, vendor_torch_root

def get_torch_cxx11_abi():
    return "1" if torch.compiled_with_cxx11_abi() else "0"

def get_torch_cmake_prefix_path():
    return torch.utils.cmake_prefix_path

def get_cmake_args():
    cmake_args = list()
    cmake_device = os.getenv("DEVICE", "")
    cmake_args.append("-DCMAKE_BUILD_TYPE=Release")
    cmake_args.append(f"-DTorch_DIR={get_torch_cmake_prefix_path()}/Torch")
    cmake_args.append(f"-D_GLIBCXX_USE_CXX11_ABI={get_torch_cxx11_abi()}")
    cmake_args.append(f"-DDEVICE={cmake_device}")
    vendor_torch_str, vendor_torch_root = get_vendor_torch_root(cmake_device)
    cmake_args.append(f"-D{vendor_torch_str.capitalize()}_ROOT={vendor_torch_root}")
    return cmake_args

def get_package_data():
    cmake_device = os.getenv("DEVICE", "").lower()
    assert cmake_device, "DEVICE shouldn't be empty!"
    return {
        f"infer_ext.vendor.{cmake_device}": [
            # f"{cmake_device}_extension.so",
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
        cmake_args=get_cmake_args(),
        cmake_install_target="install",
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
