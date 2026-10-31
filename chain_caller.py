import numpy as np
from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parent

sys.path.insert(0, str(project_root / "cpp" / "chain"))
sys.path.insert(0, str(project_root / "cu" / "chain"))
sys.path.insert(0, str(project_root / "cpp" / "decompress"))
sys.path.insert(0, str(project_root / "cu" / "decompress"))
sys.path.insert(0, str(project_root / "lcp"))
sys.path.insert(0, str(project_root / "draco_ids"))

import lcps_cpp
import lcps_gpu_cpp
import lcps_gpu_decompress_ans
import lcps_cpu_decompress
import lcps
import draco_ids


def chain(coords, eb, p, threshold, diff, bl=True, l1=True, l2=True, ec=True):
    ret = lcps_cpp.lcps_compress2(coords, eb, p, threshold, diff, bl, l1, l2, ec)
    return ret


def chain_decompress(compressed_data, timer=False):
    ret = lcps_cpu_decompress.lcps_decompress_serial(compressed_data)
    return ret


def gpu_decompress(compressed_data, timer=False):
    ret = lcps_gpu_decompress_ans.lcps_decompress_gpu(compressed_data)
    return ret


def chain_gpu(coords, eb, p, threshold, diff, timer=False):
    ret = lcps_gpu_cpp.lcps_compress2_gpu(coords, eb, p, threshold, diff)
    return ret


def lcps_compress(coords, eb, reorder=False):
    return lcps.lcps(eb, coords, reorder)


def lcps_decompress(compressed):
    return lcps.lcps_decompress(compressed)


def dracoIds(data, qb):
    return draco_ids.encode(data, quantization_bits=qb)
