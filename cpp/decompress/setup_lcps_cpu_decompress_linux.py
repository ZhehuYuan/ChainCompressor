from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Linux version for building lcps_cpu_decompress with pybind11 + zstd.
# Assumes zstd headers/libs are installed system-wide, e.g.:
#   Ubuntu/Debian: sudo apt install libzstd-dev
#   Conda:         conda install -c conda-forge zstd
# If zstd is installed in a custom prefix, set ZSTD_ROOT below.

MODULE_NAME = "lcps_cpu_decompress"
CPP_SOURCE = "lcps_cpu_decompress_serial.cpp"

ZSTD_ROOT = None  # Example: Path("/home/zyuan/local/zstd")

include_dirs = []
library_dirs = []

if ZSTD_ROOT is not None:
    ZSTD_ROOT = Path(ZSTD_ROOT)
    include_dirs.append(str(ZSTD_ROOT / "include"))
    library_dirs.append(str(ZSTD_ROOT / "lib"))

ext_modules = [
    Pybind11Extension(
        MODULE_NAME,
        [CPP_SOURCE],
        cxx_std=17,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["zstd"],
        extra_compile_args=[
            "-O3",
        ],
    ),
]

setup(
    name=MODULE_NAME,
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

