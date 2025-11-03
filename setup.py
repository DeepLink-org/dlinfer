import os
from pathlib import Path
from setuptools import find_packages
import yaml
from skbuild import setup


VERSION = "0.2.3"

vendor_dispatch_key_map = {
    "ascend": "PrivateUse1",
    "maca": "CUDA",
    "camb": "PrivateUse1",
}


def gen_vendor_yaml(device):
    config = dict()
    config["vendor"] = device
    assert device in vendor_dispatch_key_map
    config["dispatch_key"] = vendor_dispatch_key_map[device]
    file_path = Path(__file__).parent / "dlinfer" / "vendor" / "vendor.yaml"
    with open(str(file_path), "w") as f:
        yaml.safe_dump(config, f)
    return str(file_path.name)


def get_device():
    device = os.getenv("DEVICE", "").lower()
    assert device in vendor_dispatch_key_map
    return device


def get_cmake_args():
    cmake_args = list()
    cmake_device = get_device()
    cmake_args.append("-DCMAKE_BUILD_TYPE=Release")
    cmake_args.append(f"-DDEVICE={cmake_device}")
    return cmake_args


def get_package_data():
    cmake_device = get_device()
    yaml_file_name = gen_vendor_yaml(cmake_device)
    assert cmake_device, "DEVICE shouldn't be empty!"
    return {
        f"dlinfer.vendor": [
            yaml_file_name,
        ]
    }


def get_readme():
    with open(str(Path(__file__).parent / "README.md"), "r", encoding="utf-8") as f:
        content = f.read()
    return content


def get_requirements(file_name):
    requirements = []
    device_req_root = Path(__file__).parent / "requirements" / get_device()
    with open(str(device_req_root / file_name), "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r"):
                other_file = line.split()[1]
                requirements.extend(get_requirements(other_file))
            else:
                requirements.append(line)
    return requirements


def get_vendor_excludes():
    device = get_device()
    exclude_vendors = [
        name for name in vendor_dispatch_key_map.keys() if name != device
    ]
    return [f"dlinfer.vendor.{name}" for name in exclude_vendors]


def get_entry_points():
    device = get_device()
    if device == "ascend":
        return {
            "torch_dynamo_backends": [
                "atbgraph = dlinfer.graph.dicp.vendor.AtbGraph:atbgraph",
            ]
        }
    else:
        return dict()


def get_install_target():
    device = get_device()
    if device == "camb":
        return "all"
    else:
        return "install"


def main():
    setup(
        name=f"dlinfer-{get_device()}",
        version=VERSION,
        description="DeepLink Inference Extension",
        long_description=get_readme(),
        long_description_content_type="text/markdown",
        url="https://github.com/DeepLink-org/dlinfer",
        packages=find_packages(exclude=get_vendor_excludes()),
        package_data=get_package_data(),
        exclude_package_data={"": ["tests/*"]},
        cmake_args=get_cmake_args(),
        cmake_install_target=get_install_target(),
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Intended Audience :: Developers",
            "Operating System :: POSIX :: Linux",
        ],
        python_requires=">=3.8, <3.12",
        setup_requires=get_requirements("build.txt"),
        install_requires=get_requirements("runtime.txt"),
        entry_points=get_entry_points(),
    )


if __name__ == "__main__":
    main()
