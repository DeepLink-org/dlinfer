from setuptools import setup, find_packages


def main():
    setup(
        name="deeplink_ext",
        version="0.0.1",
        url="https://github.com/DeepLink-org/DeepLinkExt",
        packages=find_packages(),
        include_package_data=True,
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
