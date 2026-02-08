import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
)

library_name = "lrn_sparseatt"


if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",
            # define TORCH_TARGET_VERSION with min version 2.10 to expose only the
            # stable API subset from torch
            # Format: [MAJ 1 byte][MIN 1 byte][PATCH 1 byte][ABI TAG 5 bytes]
            # 2.10.0 = 0x020A000000000000
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    # this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(library_name, "extension")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    ext_modules = [
        CppExtension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch>=2.10.0", "numpy"],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
