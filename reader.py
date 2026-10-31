from netCDF4 import Dataset
import re
import warnings
import numpy as np
from pathlib import Path
import MDAnalysis as mda
import math
import sys
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import subprocess
import time
import chain_caller
import random
import blosc2
import pyvista as pv
from plyfile import PlyData
from pysz import sz, szConfig, szErrorBoundMode
import DracoPy
import zstandard as zstd


def vtpReader(path: str, id: int) -> np.ndarray:
    import xml.etree.ElementTree as ET

    folder = Path(path)

    prefix = f"{id:04d}"
    files = sorted(folder.glob(f"{prefix}*.vtp"))

    if len(files) == 0:
        raise FileNotFoundError(f"No .vtp file found with prefix {prefix} in {folder}")

    if len(files) > 1:
        raise RuntimeError(f"Multiple .vtp files found with prefix {prefix}: {files}")

    filename = files[0]

    tree = ET.parse(filename)
    root = tree.getroot()

    for da in root.iter("DataArray"):
        if da.attrib.get("Name") == "Points":
            ncomp = int(da.attrib.get("NumberOfComponents", 1))

            if ncomp != 3:
                raise ValueError(
                    f"{filename}: DataArray 'Points' has NumberOfComponents={ncomp}, not 3"
                )

            arr = np.fromstring(da.text, sep=" ", dtype=np.float32)

            if arr.size % 3 != 0:
                raise ValueError(
                    f"{filename}: Points array size {arr.size} is not divisible by 3"
                )

            return arr.reshape(-1, 3)

    raise ValueError(f"{filename}: no DataArray with Name='Points' found")


def callSZ3(coords, bound=10.0, timer=False):
    # SZ3
    config = szConfig()
    config.errorBoundMode = szErrorBoundMode.ABS
    # config.absErrorBound = 1e-2 / 10 * bound
    config.absErrorBound = bound
    if timer:
        start = time.perf_counter()
    compressed, _ = sz.compress(coords, config)
    if timer:
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        print("SZ3 Time:", elapsed_ms)
    return compressed


def callChainCPPTraverse(coords, p_ub, bound):
    # Tuning Parameter
    dim = coords.shape[1]
    eb = bound
    minSize = dim * len(coords) * 32
    outStream = b""
    best = []
    for p in [2 ** i for i in range(0, p_ub+1)]:
        threshold = 1
        last = 0
        while threshold < coords.shape[0].bit_length() * 3: 
            out = chain_caller.chain(coords, eb, p, threshold, 0)
            num_chained = out[-1]
            size2 = len(out[0])
            # print(size2, p, threshold, num_chained, out[1:])
            if size2 < minSize:
                minSize = size2
                minSetting = [p, threshold, num_chained]
                outStream = out[0]
            if num_chained <= 0:
                break
            threshold += 1
    # print(minSize, minSetting)
    return minSize, minSetting[0], minSetting[1], outStream


def callChainCPPAblation(coords, p_ub, bound):
    dim = coords.shape[1]
    eb = bound
    forceEC=True
    if not forceEC:
        print("L1")
        minSize = dim * len(coords) * 32
        threshold = 1
        while threshold < coords.shape[0].bit_length() * 3: 
            out = chain_caller.chain(coords, eb, 1, threshold, 0, bl=False, l1=True, l2=False, ec=False)
            size2 = len(out[0])
            minSize = min(size2, minSize)
            threshold += 1
        print("\t" + str(minSize))
        print("L2")
        minSize = dim * len(coords) * 32
        out = chain_caller.chain(coords, eb, 1, 1, 0, bl=False, l1=False, l2=True, ec=False)
        size2 = len(out[0])
        minSize = min(size2, minSize)
        print("\t" + str(minSize))
    print("Encoding")
    minSize = dim * len(coords) * 32
    out = chain_caller.chain(coords, eb, 1, 1, 0, bl=False, l1=False, l2=False, ec=True)
    size2 = len(out[0])
    minSize = min(size2, minSize)
    print("\t" + str(minSize))
    if not forceEC:
        print("L1 + 2")
        minSize = dim * len(coords) * 32
        threshold = 1
        while threshold < coords.shape[0].bit_length() * 3: 
            out = chain_caller.chain(coords, eb, 1, threshold, 0, bl=False, l1=True, l2=True, ec=False)
            size2 = len(out[0])
            minSize = min(size2, minSize)
            threshold += 1
        print("\t" + str(minSize))
    print("L1 + Encoding")
    minSize = dim * len(coords) * 32
    threshold = 1
    while threshold < coords.shape[0].bit_length() * 3: 
        out = chain_caller.chain(coords, eb, 1, threshold, 0, bl=False, l1=True, l2=False, ec=True)
        size2 = len(out[0])
        minSize = min(size2, minSize)
        threshold += 1
    print("\t" + str(minSize))
    print("L2 + Encoding")
    minSize = dim * len(coords) * 32
    out = chain_caller.chain(coords, eb, 1, 1, 0, bl=False, l1=False, l2=True, ec=True)
    size2 = len(out[0])
    minSize = min(size2, minSize)
    print("\t" + str(minSize))
    if not forceEC:
        print("Block + L1")
        minSize = dim * len(coords) * 32
        for p in [2 ** i for i in range(0, p_ub+1)]:
            threshold = 1
            while threshold < coords.shape[0].bit_length() * 3: 
                out = chain_caller.chain(coords, eb, p, threshold, 0, bl=True, l1=True, l2=False, ec=False)
                size2 = len(out[0])
                if size2 < minSize:
                    minSize = size2
                threshold += 1
        print("\t" + str(minSize))
        print("Block + L2")
        minSize = dim * len(coords) * 32
        for p in [2 ** i for i in range(0, p_ub+1)]:
            out = chain_caller.chain(coords, eb, p, 1, 0, bl=True, l1=False, l2=True, ec=False)
            size2 = len(out[0])
            if size2 < minSize:
                minSize = size2
        print("\t" + str(minSize))
    print("Block + Encoding")
    minSize = dim * len(coords) * 32
    for p in [2 ** i for i in range(0, p_ub+1)]:
        out = chain_caller.chain(coords, eb, p, 1, 0, bl=True, l1=False, l2=False, ec=True)
        size2 = len(out[0])
        if size2 < minSize:
            minSize = size2
    print("\t" + str(minSize))
    print("Block + L2 + Encoding")
    minSize = dim * len(coords) * 32
    for p in [2 ** i for i in range(0, p_ub+1)]:
        out = chain_caller.chain(coords, eb, p, 1, 0, bl=True, l1=False, l2=True, ec=True)
        size2 = len(out[0])
        if size2 < minSize:
            minSize = size2
    print("\t" + str(minSize))
    print("Block + L1 + Encoding")
    minSize = dim * len(coords) * 32
    for p in [2 ** i for i in range(0, p_ub+1)]:
        threshold = 1
        while threshold < coords.shape[0].bit_length() * 3: 
            out = chain_caller.chain(coords, eb, p, threshold, 0, bl=True, l1=True, l2=False, ec=True)
            size2 = len(out[0])
            if size2 < minSize:
                minSize = size2
            threshold += 1
    print("\t" + str(minSize))
    if not forceEC:
        print("Block + L1 + 2")
        minSize = dim * len(coords) * 32
        for p in [2 ** i for i in range(0, p_ub+1)]:
            threshold = 1
            while threshold < coords.shape[0].bit_length() * 3: 
                out = chain_caller.chain(coords, eb, p, threshold, 0, bl=True, l1=True, l2=True, ec=False)
                size2 = len(out[0])
                if size2 < minSize:
                    minSize = size2
                threshold += 1
        print("\t" + str(minSize))
    print("L1 + 2 + Encoding")
    minSize = dim * len(coords) * 32
    threshold = 1
    while threshold < coords.shape[0].bit_length() * 3: 
        out = chain_caller.chain(coords, eb, 1, threshold, 0, bl=False, l1=True, l2=True, ec=True)
        size2 = len(out[0])
        minSize = min(size2, minSize)
        threshold += 1
    print("\t" + str(minSize))


def callChainCUDATraverse(coords, p_ub, bound):
    # Tuning Parameter
    coordsT = coords.T
    dim = coords.shape[1]
    eb = bound
    totalTime = 0.0
    minSize = dim * len(coords) * 32
    outStream = b""
    for p in [2 ** i for i in range(0, p_ub+1)]:
        threshold = 2
        while threshold < coords.shape[0].bit_length():
                out = chain_caller.chain_gpu(coordsT, eb, p, threshold, 0)
                tmpOutStream = out
                size = len(tmpOutStream)
                # print(size, p, threshold, out[-1])
                if size < minSize:
                    minSize = size
                    minSetting = (p, threshold, diff, out[-1])
                    outStream = tmpOutStream
                threshold += 1
    # print(minSize, minSetting)
    return minSize, outStream


def callCUDADecompress(data, timer=False):
    data = chain_caller.gpu_decompress(data, timer)
    return data


def callDraco(coords, qb=10, byte=False):
        qb = qb - 1
        binary = DracoPy.encode(
            coords,
            quantization_bits=qb,
            compression_level=10,
            preserve_order=True
        )
        if byte:
            return binary
        return len(binary)


def equivalent_eb(bounds, qb):
    # Draco only support quantization bit(qb)
    # We can use this function to transform qb to error bound
    return np.max(bounds) / (2 **(qb + 1) - 2)


def realPSNR(original_data, decompressed_data):
    original = np.asarray(original_data, dtype=np.float64)
    decompressed = np.asarray(decompressed_data, dtype=np.float64)

    if original.shape != decompressed.shape:
        raise ValueError(
            f"Shape mismatch: {original.shape} != {decompressed.shape}"
        )

    dimension_ranges = original.max(axis=0) - original.min(axis=0)
    data_range = dimension_ranges.max()

    mse = np.mean((original - decompressed) ** 2)

    if mse == 0:
        return np.inf

    if data_range == 0:
        return -np.inf

    return 20.0 * np.log10(data_range / np.sqrt(mse))


def testCPU():
            # Dam Break
            ts = [4, 25, 22, 25] 
            ps = [256, 4, 16, 16]
            qbs = [11, 12, 13, 14]
            print("Dam Break")
            data = vtpReader(r"./data/DamBreak/", 654)
            size = len(data)
            maximum = data.max(axis=0) - data.min(axis=0)
            for qb in qbs:
                eb = equivalent_eb(maximum, qb-1)
                print("qb: " + str(qb))
                encoded, ids = chain_caller.dracoIds(data, qb-1)
                decoded = DracoPy.decode(encoded).points
                decoded = decoded[ids]
                draco = len(encoded) * 8
                print(draco/size, realPSNR(data, decoded))
                encoded = callDraco(data, qb=qb, byte=True)
                decoded = DracoPy.decode(encoded).points
                draco = len(encoded) * 8
                print(draco/size, realPSNR(data, decoded))
                encoded = callSZ3(data, bound = eb)
                decoded, _ = sz.decompress(encoded, np.float32, data.shape)
                decoded = decoded.reshape(-1, 3)
                sz3 = encoded.nbytes * 8
                print(sz3/size, realPSNR(data, decoded))
                encoded, _ = chain_caller.lcps_compress(data, eb, False)
                decoded = chain_caller.lcps_decompress(encoded)
                lcp = len(encoded) * 8
                print(lcp/size, realPSNR(data, decoded))
                encoded, ids = chain_caller.lcps_compress(data, eb, True)
                decoded = chain_caller.lcps_decompress(encoded)
                compressed_ids = zstd.ZstdCompressor(level=22).compress(ids.tobytes())
                lcp_order = (len(encoded) + len(compressed_ids)) * 8
                restored = np.empty_like(decoded)
                restored[ids] = decoded
                print([lcp_order/size, len(encoded) * 8 / size], realPSNR(data, restored))
                encoded = chain_caller.chain(data, eb, ps[qb-11], ts[qb-11], 0)[0]
                decoded = chain_caller.chain_decompress(encoded)
                print(len(encoded)*8/size, realPSNR(data, decoded))

def testGPU():
      qbs = [11, 12, 13, 14]
      print("Dam Break")
      random.seed(42)
      ts = [2, 2, 16, 2]
      ps = [2, 4, 16, 16]
      data = vtpReader(r"./data/DamBreak/", 654)
      size = len(data)
      maximum = data.max(axis=0) - data.min(axis=0)
      for qb in qbs:
        eb = equivalent_eb(maximum, qb-1)
        print("qb: " + str(qb))
        encoded = chain_caller.chain_gpu(data.T, eb, ps[qb-11], ts[qb-11], 0)
        decoded = chain_caller.gpu_decompress(encoded)


def testAblation():
    random.seed(42)
    ts = [4, 25, 22, 25]
    ps = [256, 4, 16, 16]
    missing = [325, 412, 450, 476, 486, 499, 550, 555, 598, 604, 648, 664, 696, 702, 793, 875, 951]
    data = vtpReader(r"./data/DamBreak/", 654)
    size = len(data)
    maximum = data.max(axis=0) - data.min(axis=0)
    eb = equivalent_eb(maximum, 10)
    callChainCPPAblation(data, 10, eb)


testCPU()
testGPU()
testAblation()
# For github space reason, we only upload the randomly selected Dam Break dataset.
# But the best parameters for each dataset under qb=11, 12, 13, ... are provided:
# CPU: 
    # CAR: ts = [1, 1, 4, 2] ps = [256, 256, 256, 256]
    # LES: ts = [1, 5, 8, 7] ps = [256, 256, 256, 256]
    # David: ts = [17, 20, 23, 26] ps = [1, 1, 1, 1]
    # sdrBench HACC: ts = [25, 28, 31, 31] ps = [1, 1, 1, 2]
    # Dark Sky: ts = [25, 23, 25, 25, 25] ps = [2, 4, 8, 16, 32]
    # NhaA: ts = [1, 1, 1, 1] ps = [2, 4, 8, 16]
    # Yiip: ts = [1, 1, 1, 1] ps = [2, 4, 8, 16]
    # Particular Ensemble: ts = [3, 3, 3, 3] ps = [32, 64, 128, 256]
    # Dam Break: ts = [4, 25, 22, 25] ps = [256, 4, 16, 16]
# GPU:
    # CAR: ts = [22, 20, 23, 22] ps = [2, 1, 1, 1] 
    # LES: ts = [14, 24, 23, 23] ps = [1, 1, 2, 4]
        # For GPU memory limitation, we test LES with its first 22M particles on GPU
    # David: ts = [19, 20, 19, 2] ps = [2, 1, 1, 1]
    # sdrBench HACC: ts = [23, 23, 23, 23] ps = [1, 1, 2, 4] 
    # Dark Sky: ts = [14, 14, 14, 14] ps = [16, 32, 64, 128]
    # Yiip: ts = [2, 2, 2, 2] ps = [1, 2, 4, 8]
    # Particular Ensemble: ts = [9, 9, 9, 9] ps = [2, 4, 8, 16] 
    # Dam Break: ts = ts = [2, 2, 16, 2], ps = [2, 4, 16, 16]
