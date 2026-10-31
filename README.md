# Chain Compressor

Chain Compressor provides order-preserving lossy compression for particle coordinates, with C++ and CUDA implementations exposed through Python bindings. This repository also includes LCP-S and Draco wrappers and an experiment script for comparison with SZ3, Draco, and LCP-S.

## Repository structure

| Path | Description |
| --- | --- |
| `reader.py` | Main experiment script: data loading, CPU comparisons, GPU tests, parameter searches, and ablation experiments. |
| `chain_caller.py` | Python wrappers for the Chain CPU/GPU implementations and the LCP-S and Draco extensions. |
| `cpp/chain/lcps_cpp.cpp` | CPU Chain compressor, including its paired decoder and ablation switches. |
| `cpp/chain/setup_linux.py` | Build script for the `lcps_cpp` extension. |
| `cpp/decompress/lcps_cpu_decompress_serial.cpp` | Standalone serial CPU decompressor. |
| `cpp/decompress/setup_lcps_cpu_decompress_linux.py` | Build script for the `lcps_cpu_decompress` extension. |
| `cu/chain/lcps_gpu_cpp.cu` | CUDA Chain compressor using nvCOMP ANS and Zstd. |
| `cu/chain/setup_lcps_gpu_linux.py` | Build script for the `lcps_gpu_cpp` extension. |
| `cu/decompress/lcps_gpu_decompress_ans.cu` | CUDA Chain decompressor. |
| `cu/decompress/setup_lcps_gpu_decompress_linux.py` | Build script for the `lcps_gpu_decompress_ans` extension. |
| `lcp/lcps.cpp` | LCP-S extension, supporting compression with or without particle reordering. |
| `lcp/Makefile` | Builds the `lcps` extension. |
| `draco_ids/draco_ids.cpp` | Draco wrapper that returns compressed bytes and an index array for restoring the original particle order. |
| `draco_ids/setup_draco_linux.py` | Builds the `draco_ids` extension from bundled Draco sources. |
| `draco_ids/_draco_ids_vendor/` | Bundled Draco sources, patches, and third-party licenses. |
| `data/DamBreak/` | One Dam Break sample in ASCII VTP format, containing 80,703 particles. |

The Chain extension names retain the `lcps_` prefix. The separate LCP-S baseline is implemented in `lcp/`.

## Requirements

- Linux with a 64-bit Python environment, Python development headers, a C++17 compiler, and GNU Make.
- Zstd development headers and libraries for the CPU Chain modules.
- An NVIDIA GPU, a compatible CUDA Toolkit with `nvcc`, and nvCOMP development headers and libraries with ANS and Zstd support for the GPU modules.

Use the same Python environment to build and run all extensions.

### Clone and install Python dependencies

```bash
git clone https://github.com/ZhehuYuan/ChainCompressor.git
cd ChainCompressor

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pybind11
python -m pip install netCDF4 MDAnalysis vtk blosc2 pyvista plyfile pysz DracoPy zstandard
```

The final command installs the additional packages imported by `reader.py`.

On Ubuntu/Debian, the CPU build prerequisites can be installed with administrator access:

```bash
sudo apt install build-essential python3-dev python3-venv libzstd-dev
```

Install the system prerequisites before creating the virtual environment if Python's `venv` support is not already available. On a cluster, use the corresponding installed packages or environment modules.

## Build the extensions

Run the following commands from the repository root. The subshells keep the working directory unchanged after each build.

Build all six extensions before running `reader.py` or importing `chain_caller.py`, which imports all six modules.

### 1. CPU Chain

Configure `ZSTD_ROOT` in both CPU build scripts for your installation. Use `None` for a system-wide Zstd installation, or `Path("/path/to/zstd")` for a custom prefix containing `include/` and `lib/`. For a custom installation, also add its library directory to `LD_LIBRARY_PATH`.

```bash
(cd cpp/chain && python setup_linux.py build_ext --inplace)
(cd cpp/decompress && python setup_lcps_cpu_decompress_linux.py build_ext --inplace)
```

### 2. GPU Chain

The GPU build scripts detect CUDA and nvCOMP from standard installation locations and supported Python/Conda locations. For custom installations, set the paths explicitly, replacing the example paths below:

```bash
export CUDA_HOME=/path/to/cuda
export NVCOMP_ROOT=/path/to/nvcomp
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NVCOMP_ROOT/lib:$NVCOMP_ROOT/lib64:${LD_LIBRARY_PATH:-}"
```

`CUDA_HOME` must contain `bin/nvcc`, and `NVCOMP_ROOT` must contain `include/nvcomp.h` and the nvCOMP libraries.

```bash
(cd cu/chain && python setup_lcps_gpu_linux.py build_ext --inplace)
(cd cu/decompress && python setup_lcps_gpu_decompress_linux.py build_ext --inplace)
```

The scripts detect the architecture of visible GPUs. When compiling on a machine without a visible GPU, set `CUDA_ARCH_LIST` to the target GPU architecture, for example `export CUDA_ARCH_LIST="80"` for an A100 or `export CUDA_ARCH_LIST="86"` for an RTX A6000, before building.

### 3. LCP-S and Draco wrappers

```bash
make -C lcp PYTHON=python
(cd draco_ids && python setup_draco_linux.py build_ext --inplace)
```

The LCP-S source includes its required compression code. The Draco build script uses its bundled sources; neither build requires CMake.

### 4. Check the imports

```bash
python -c "import chain_caller; print('All extensions imported successfully.')"
```

## Run the bundled experiments

From the repository root, run:

```bash
python reader.py
```

The script loads frame `0654` from `data/DamBreak/` and runs these functions in order:

| Function | Experiment |
| --- | --- |
| `testCPU()` | Compares Draco variants, SZ3, LCP-S variants, and CPU Chain, reporting bit rates and PSNR. |
| `testGPU()` | Runs GPU Chain compression and decompression for four parameter settings and prints stage timings. |
| `testAblation()` | Reports compressed sizes for selected combinations of block partitioning, chain construction, and encoding. |

To run only selected experiments, keep the corresponding calls at the end of `reader.py` and comment out the others.

Only one Dam Break sample is included to keep the repository small. CPU/GPU parameter settings for other datasets are listed in the comments at the end of `reader.py`.

## Use Chain from Python

After building all extensions, run this example from the repository root:

```python
import numpy as np
import chain_caller

# Particle coordinates: one point per row, columns x/y/z.
coords = np.random.default_rng(42).random((1000, 3), dtype=np.float32)
eb = 1e-3
p = 16
threshold = 2
diff = 0

# CPU compression returns a list: compressed bytes, then component statistics.
cpu_result = chain_caller.chain(coords, eb, p, threshold, diff)
cpu_bytes = cpu_result[0]
cpu_decoded = chain_caller.chain_decompress(cpu_bytes)
print("CPU:", len(cpu_bytes), "bytes; decoded shape:", cpu_decoded.shape)

# GPU compression takes coordinates in (D, N) layout and returns bytes directly.
gpu_input = np.ascontiguousarray(coords.T)
gpu_bytes = chain_caller.chain_gpu(gpu_input, eb, p, threshold, diff)
gpu_decoded = chain_caller.gpu_decompress(gpu_bytes)
print("GPU:", len(gpu_bytes), "bytes; decoded shape:", gpu_decoded.shape)
```

Both decoded arrays use the original particle order and have shape `(N, D)` by default. CPU decoding returns `float64`; the current GPU format returns `float32`. Use the matching CPU or GPU decompressor for each compressed stream.

| Parameter | Meaning |
| --- | --- |
| `eb` | Positive absolute-error parameter used for coordinate quantization. |
| `p` | Positive block size in quantized coordinates. |
| `threshold` | Parameter controlling the decision to use a relative reference during level-1 chain construction. |
| `diff` | Legacy argument; use `0` with the current implementation. |

The example uses illustrative parameters. The bundled experiments use the dataset-specific settings in `reader.py`.
