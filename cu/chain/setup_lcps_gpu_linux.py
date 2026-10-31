from __future__ import annotations

import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import List, Optional, Sequence

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


MODULE_NAME = "lcps_gpu_cpp"
CUDA_SOURCE = os.environ.get("LCPS_GPU_SOURCE", "lcps_gpu_cpp.cu")


def _version_key(path: Path) -> tuple:
    """Sort CUDA folders such as v12.2 numerically rather than lexicographically."""
    nums = re.findall(r"\d+", path.name)
    return tuple(int(x) for x in nums) if nums else (0,)


def find_cuda_home() -> Optional[Path]:
    env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "bin" / "nvcc").exists():
            return p

    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        return Path(nvcc_on_path).resolve().parent.parent

    candidates = [Path("/usr/local/cuda"), Path("/opt/cuda")]
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*"), key=_version_key, reverse=True))
    candidates.extend(sorted(Path("/opt").glob("cuda-*"), key=_version_key, reverse=True))

    for p in candidates:
        if (p / "bin" / "nvcc").exists():
            return p.resolve()
    return None


def find_nvcomp_root() -> Optional[Path]:
    env = os.environ.get("NVCOMP_ROOT") or os.environ.get("NVCOMP_HOME")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "include" / "nvcomp.h").exists():
            return p

    candidates: List[Path] = []

    # PyPI installation, e.g. nvidia-libnvcomp-cu12.
    for key in ("purelib", "platlib"):
        site_dir = sysconfig.get_paths().get(key)
        if site_dir:
            candidates.append(Path(site_dir) / "nvidia" / "libnvcomp")
            candidates.append(Path(site_dir) / "nvidia" / "nvcomp")

    # Conda installations often place headers directly under $CONDA_PREFIX/include.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix))

    candidates.extend(
        [
            Path("/usr/local/nvcomp"),
            Path("/opt/nvcomp"),
            Path("/usr/local"),
            Path("/usr"),
            Path("/usr/local/cuda"),
        ]
    )

    for p in candidates:
        if (p / "include" / "nvcomp.h").exists():
            return p.resolve()
    return None


def existing_library_dirs(root: Path) -> List[Path]:
    candidates = [
        root / "lib64",
        root / "lib",
        root / "lib" / "x86_64-linux-gnu",
        root / "targets" / "x86_64-linux" / "lib",
    ]
    return [p.resolve() for p in candidates if p.exists()]


def _normalize_arch(token: str) -> str:
    token = token.strip().lower()
    token = token.replace("sm_", "").replace("compute_", "").replace(".", "")
    if not token.isdigit() or len(token) not in (2, 3):
        raise ValueError(
            "Invalid CUDA architecture {!r}. Use values such as 80, 86, 89, or 90.".format(token)
        )
    return token


def detect_visible_gpu_arches() -> List[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    try:
        output = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    arches = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            arches.append(_normalize_arch(line))
        except ValueError:
            continue
    return sorted(set(arches), key=int)


def cuda_arch_flags() -> List[str]:
    """
    CUDA_ARCH_LIST examples:
      export CUDA_ARCH_LIST="80"          # A100
      export CUDA_ARCH_LIST="86;89"       # Ampere + Ada
      export CUDA_ARCH_LIST="native"      # visible build-host GPUs only

    When unset, visible GPU compute capabilities are detected with nvidia-smi.
    If no GPU is visible (common on cluster login nodes), CUDA_ARCH_FALLBACK is
    used; its default is 80, appropriate for A100. Override it when necessary.
    """
    requested = os.environ.get("CUDA_ARCH_LIST", "").strip()
    if requested.lower() == "native":
        return ["-arch=native"]

    if requested:
        tokens = re.split(r"[;,\s]+", requested)
        arches = sorted({_normalize_arch(x) for x in tokens if x}, key=int)
    else:
        arches = detect_visible_gpu_arches()

    if not arches:
        fallback = os.environ.get("CUDA_ARCH_FALLBACK", "80")
        arches = sorted(
            {_normalize_arch(x) for x in re.split(r"[;,\s]+", fallback) if x},
            key=int,
        )
        print(
            "WARNING: no visible GPU architecture was detected; "
            "using CUDA_ARCH_FALLBACK={}. Set CUDA_ARCH_LIST explicitly if needed.".format(
                ";".join(arches)
            )
        )

    flags: List[str] = []
    for arch in arches:
        flags.append("--generate-code=arch=compute_{0},code=sm_{0}".format(arch))

    # Retain PTX for the highest requested architecture for forward compatibility.
    highest = arches[-1]
    flags.append("--generate-code=arch=compute_{0},code=compute_{0}".format(highest))
    return flags


class CUDABuildExt(build_ext):
    def build_extension(self, ext: Extension) -> None:
        cuda_sources = [s for s in ext.sources if s.endswith(".cu")]
        other_sources = [s for s in ext.sources if not s.endswith(".cu")]

        if cuda_sources:
            cuda_home = find_cuda_home()
            if cuda_home is None:
                raise RuntimeError(
                    "CUDA Toolkit was not found. Set CUDA_HOME to the directory containing bin/nvcc."
                )

            nvcc = cuda_home / "bin" / "nvcc"
            nvcomp_root = find_nvcomp_root()
            if nvcomp_root is None:
                raise RuntimeError(
                    "nvCOMP was not found. Install the nvCOMP development package or set "
                    "NVCOMP_ROOT to a directory containing include/nvcomp.h."
                )

            build_temp = Path(self.build_temp)
            build_temp.mkdir(parents=True, exist_ok=True)

            include_dirs = list(ext.include_dirs or [])
            include_dirs.extend(
                [
                    sysconfig.get_paths()["include"],
                    pybind11.get_include(),
                    str(cuda_home / "include"),
                    str(nvcomp_root / "include"),
                ]
            )

            arch_flags = cuda_arch_flags()
            objects: List[str] = []

            for source in cuda_sources:
                source_path = Path(source)
                if not source_path.exists():
                    raise FileNotFoundError(
                        "CUDA source not found: {}. Rename the file or set LCPS_GPU_SOURCE.".format(
                            source_path
                        )
                    )

                object_path = build_temp / (source_path.stem + ".o")
                command = [
                    str(nvcc),
                    "-c",
                    str(source_path),
                    "-o",
                    str(object_path),
                    "-std=c++17",
                    "-O3",
                    "-DNDEBUG",
                    "--expt-relaxed-constexpr",
                    "-Xcompiler",
                    "-fPIC",
                ]
                command.extend(arch_flags)
                for include_dir in dict.fromkeys(include_dirs):
                    command.extend(["-I", str(include_dir)])

                print("Compiling CUDA source:\n  " + " ".join(command))
                subprocess.check_call(command)
                objects.append(str(object_path))

            ext.sources = other_sources
            ext.extra_objects = list(ext.extra_objects or []) + objects

        super().build_extension(ext)


cuda_home = find_cuda_home()
if cuda_home is None:
    raise RuntimeError(
        "CUDA Toolkit was not found. Set CUDA_HOME to the directory containing bin/nvcc."
    )

nvcomp_root = find_nvcomp_root()
if nvcomp_root is None:
    raise RuntimeError(
        "nvCOMP was not found. Install the nvCOMP development package or set "
        "NVCOMP_ROOT to a directory containing include/nvcomp.h."
    )

include_dirs = [
    pybind11.get_include(),
    str(cuda_home / "include"),
    str(nvcomp_root / "include"),
]

library_paths: List[Path] = []
library_paths.extend(existing_library_dirs(cuda_home))
library_paths.extend(existing_library_dirs(nvcomp_root))
# Preserve order while removing duplicate resolved paths.
library_paths = list(dict.fromkeys(library_paths))

if not library_paths:
    raise RuntimeError("No CUDA/nvCOMP library directories were found.")

library_dirs = [str(p) for p in library_paths]

# RUNPATH is needed when nvCOMP comes from a custom prefix or a PyPI package.
extra_link_args = ["-Wl,--enable-new-dtags", "-Wl,-rpath,$ORIGIN"]
extra_link_args.extend("-Wl,-rpath,{}".format(path) for path in library_dirs)

extension = Extension(
    name=MODULE_NAME,
    sources=[CUDA_SOURCE],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["cudart", "nvcomp"],
    language="c++",
    extra_compile_args=["-O3", "-std=c++17"],
    extra_link_args=extra_link_args,
)

setup(
    name=MODULE_NAME,
    version="0.1.0",
    ext_modules=[extension],
    cmdclass={"build_ext": CUDABuildExt},
)
