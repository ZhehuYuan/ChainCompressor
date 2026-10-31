// Serial host orchestration + GPU transforms and batched nvCOMP compression.
// Build: C++17, nvCOMP with ANS and Zstd support, normal IEEE float arithmetic.
// Input: float32 SoA (D, N). Return: one self-contained LCG2 bytes object.
//
// LCG2 version 1 (little-endian; no native struct padding):
//   magic[4]="LCG2", u32 version=1, u32 D, u64 N, i32 p, f32 eb,
//   i32 wrap_min, i32 wrap_max, f32 minima[D], u32 stream_count=5.
//   Five entries, in order IDR0/DIF0/KEP0/CH20/REL0:
//     name[4], u64 component_bytes, component.
//   Component:
//     u8 encoding (0=empty, 1=Tagged+Zstd, 2=Fixed+Zstd, 3=Raw+ANS),
//     u32 D, u64 records,
//     if Fixed: D * (i32 minimum, u8 width, u64 packed_dimension_bytes),
//     if Tagged: u64 valid_packed_bits,
//     u8 section_count, sections.
//   Section:
//     u8 codec (1=nvCOMP rANS/UCHAR, 2=nvCOMP Zstd), u64 raw_bytes,
//     u32 chunk_count, chunk_count * (u64 raw_chunk_bytes, u64 encoded_chunk_bytes),
//     encoded chunks concatenated in that order.
//   Fixed: one Zstd section; MSB-first per-axis offsets, each axis byte padded.
//   Tagged: ANS section for one uint8 tag per record, then Zstd section for
//           MSB-first point-major values using that record's shared tag.
//   Raw: one ANS section for point-major little-endian int32 bytes (4/value).
//   Empty: records=0, section_count=0. No other fields or codec frames.
//
// Reconstruction (dedicated GPU decoder, NOT the CPU LCP2 decoder):
//   choices=CH20; reconstruct compact blocks from KEP0 and DIF0 in their order.
//   DIF0 symbols = ZigZag(wrap(previous_block - current_block)); invert modulo
//   (wrap_max-wrap_min+1). The first compact block always comes from KEP0.
//   IDR0 has N entries: 0 consumes an explicit compact block; a positive value
//   references a successor at i+IDR0[i]. Follow those references backward.
//   q = block*p + REL0, using exact integer arithmetic.
//   coordinate = fmaf(float(q)+0.5f, 2.0f*eb, minima[d]) in round-to-nearest.
//   There are no exact-coordinate corrections. Additional error is restricted
//   to normal float32 rounding; all post-quantization transforms are lossless.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#if __has_include(<nvcomp/ans.h>)
#include <nvcomp/ans.h>
#include <nvcomp/zstd.h>
#elif __has_include(<ans.h>)
#include <ans.h>
#include <zstd.h>
#else
#error "nvCOMP ANS and Zstd headers are required"
#endif
#include <cub/cub.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__FAST_MATH__) || defined(__CUDA_FAST_MATH__)
#error "LCG2 float quantization requires compilation without fast-math"
#endif
#if defined(__CUDA_FTZ) && __CUDA_FTZ
#error "LCG2 float quantization requires --ftz=false"
#endif

namespace py = pybind11;
namespace lcps_gpu {

using I32 = int;
using U64 = unsigned long long;
static_assert(sizeof(I32) == 4 && sizeof(float) == 4, "LCG2 requires int32/float32");
constexpr int kThreads = 256;
constexpr int kStreamCount = 5;
constexpr std::size_t kAnsChunkBytes = 1u << 24;
constexpr std::size_t kZstdChunkBytes = 65536;
constexpr int kRequiredSavingBits = 0;
constexpr const char* kStreamNames[kStreamCount] = {"IDR0", "DIF0", "KEP0", "CH20", "REL0"};

static void require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
static std::size_t checked_add(std::size_t a, std::size_t b) {
    require(b <= std::numeric_limits<std::size_t>::max() - a, "LCG2 size overflow");
    return a + b;
}
static std::size_t checked_mul(std::size_t a, std::size_t b) {
    require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b, "LCG2 size overflow");
    return a * b;
}
static std::size_t ceil_div(std::size_t n, std::size_t d) { return n/d + (n%d != 0); }
static std::size_t align_up(std::size_t n, std::size_t alignment) {
    return checked_mul(ceil_div(n, alignment), alignment);
}
static int grid_for(std::size_t n) {
    return static_cast<int>(std::min<std::size_t>(65535, std::max<std::size_t>(1, ceil_div(n, kThreads))));
}
static void check_cuda(cudaError_t status, const char* expression, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status)
                                 + " (line " + std::to_string(line) + ")");
    }
}
static void check_nvcomp(nvcompStatus_t status, const char* expression, int line) {
    if (status != nvcompSuccess) {
        throw std::runtime_error(std::string(expression) + ": nvCOMP status "
                                 + std::to_string(static_cast<int>(status))
                                 + " (line " + std::to_string(line) + ")");
    }
}
#define LCPS_CUDA(expr) ::lcps_gpu::check_cuda((expr), #expr, __LINE__)
#define LCPS_NVCOMP(expr) ::lcps_gpu::check_nvcomp((expr), #expr, __LINE__)

template<class T> struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t count = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { allocate(n); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), count(other.count) {
        other.ptr = nullptr; other.count = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        std::swap(ptr, other.ptr); std::swap(count, other.count); return *this;
    }
    void allocate(std::size_t n) {
        require(ptr == nullptr, "device buffers cannot be reallocated during compression");
        count = n;
        if (n) LCPS_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), checked_mul(n, sizeof(T))));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
};

struct HostStaging {
    unsigned char* ptr = nullptr;
    explicit HostStaging(std::size_t bytes) {
        LCPS_CUDA(cudaMallocHost(reinterpret_cast<void**>(&ptr), std::max<std::size_t>(1, bytes)));
    }
    HostStaging(const HostStaging&) = delete;
    HostStaging& operator=(const HostStaging&) = delete;
    ~HostStaging() { if (ptr) cudaFreeHost(ptr); }
};

// BEGIN LCG2_INTEGER_RULES: shared host/device rules, also usable in reference tests.
__host__ __device__ inline int bit_width(std::uint32_t value) {
#ifdef __CUDA_ARCH__
    return value ? 32 - __clz(value) : 0;
#else
    int bits = 0;
    while (value) { ++bits; value >>= 1; }
    return bits;
#endif
}
__host__ __device__ inline U64 tagged_cost(std::uint32_t maximum, int dims) {
    const int width = bit_width(maximum);
    return U64(dims) * width + bit_width(static_cast<std::uint32_t>(width));
}
__host__ __device__ inline bool use_relative(
    I32 distance, std::uint32_t local_max, std::uint32_t global_max, int dims, I32 threshold
) {
    if (distance <= 0) return false;
    const U64 tagged = tagged_cost(local_max, dims);
    const U64 fixed = U64(dims) * bit_width(global_max);
    const long long block_cost = static_cast<long long>(tagged < fixed ? tagged : fixed);
    const long long relative_cost = static_cast<long long>(tagged_cost(distance, 1));
    return relative_cost < block_cost - static_cast<long long>(threshold);
}
struct BlockStats { I32 minimum; I32 maximum; std::uint32_t maximum_difference; };
struct Wrap {
    I32 minimum = 0, maximum = 0;
    std::uint32_t range = 1;
    I32 min_correction = 0, max_correction = 0;
};
static Wrap make_wrap(const BlockStats& s) {
    require(s.minimum >= 0 && s.minimum <= s.maximum, "invalid compact block range");
    Wrap w;
    w.minimum = s.minimum; w.maximum = s.maximum;
    w.range = static_cast<std::uint32_t>(s.maximum) - static_cast<std::uint32_t>(s.minimum) + 1u;
    w.min_correction = -static_cast<I32>(w.range / 2);
    w.max_correction = static_cast<I32>(w.range / 2) - ((w.range & 1u) == 0);
    return w;
}
__host__ __device__ inline bool use_difference(
    std::uint32_t local_max, std::uint32_t local_difference, const BlockStats& global, int dims
) {
    const U64 tag_value = tagged_cost(local_max, dims);
    const U64 fix_value = U64(dims) * bit_width(static_cast<std::uint32_t>(global.maximum));
    const U64 tag_diff = tagged_cost(local_difference, dims);
    const U64 fix_diff = U64(dims) * bit_width(global.maximum_difference);
    const U64 value_cost = tag_value < fix_value ? tag_value : fix_value;
    const U64 difference_cost = tag_diff > fix_diff ? tag_diff : fix_diff;
    return value_cost >= difference_cost + kRequiredSavingBits;
}
__host__ __device__ inline I32 wrapped_symbol(I32 previous, I32 current, const Wrap& w) {
    // Both values came from the reduced block range; clamping the prediction
    // to [minimum, maximum] is therefore an identity operation.
    // Nonnegative int32 inputs make this signed subtraction safe.
    I32 correction = previous - current;
    if (correction < w.min_correction) {
        correction = static_cast<I32>(w.range - static_cast<std::uint32_t>(-correction));
    } else if (correction > w.max_correction) {
        correction = -static_cast<I32>(w.range - static_cast<std::uint32_t>(correction));
    }
    const std::uint32_t symbol = correction >= 0
        ? static_cast<std::uint32_t>(correction) * 2u
        : static_cast<std::uint32_t>(-(correction + 1)) * 2u + 1u;
    return static_cast<I32>(symbol);  // symbol <= range-1 <= INT32_MAX
}
// END LCG2_INTEGER_RULES

struct Matrix { const I32* data = nullptr; int dims = 0, stride = 0, n = 0; };
struct ReadBlockStats {
    Matrix input;
    bool differences;
    __host__ __device__ BlockStats operator()(I32 i) const {
        BlockStats s{INT_MAX, 0, 0};
        for (int d = 0; d < input.dims; ++d) {
            const I32 v = input.data[std::size_t(d)*input.stride+i];
            s.minimum = v < s.minimum ? v : s.minimum;
            s.maximum = v > s.maximum ? v : s.maximum;
            if (differences && i > 0) {
                const I32 prev = input.data[std::size_t(d)*input.stride+i-1];
                const auto delta = static_cast<std::uint32_t>(v >= prev ? v-prev : prev-v);
                s.maximum_difference = delta > s.maximum_difference ? delta : s.maximum_difference;
            }
        }
        return s;
    }
};
struct MergeBlockStats {
    __host__ __device__ BlockStats operator()(const BlockStats& a, const BlockStats& b) const {
        return {a.minimum < b.minimum ? a.minimum : b.minimum,
                a.maximum > b.maximum ? a.maximum : b.maximum,
                a.maximum_difference > b.maximum_difference ? a.maximum_difference : b.maximum_difference};
    }
};
using CountIterator = cub::CountingInputIterator<I32>;
using BlockStatsIterator = cub::TransformInputIterator<BlockStats, ReadBlockStats, CountIterator>;

struct MinMax { I32 minimum, maximum; };
struct ReadMinMax {
    __host__ __device__ MinMax operator()(I32 value) const { return {value, value}; }
};
struct MergeMinMax {
    __host__ __device__ MinMax operator()(const MinMax& a, const MinMax& b) const {
        return {a.minimum < b.minimum ? a.minimum : b.minimum,
                a.maximum > b.maximum ? a.maximum : b.maximum};
    }
};
using MinMaxIterator = cub::TransformInputIterator<MinMax, ReadMinMax, const I32*>;

__global__ void quantize_bid_rel_kernel(
    const float* points, const float* minima, int n, int dims, float eb, I32 p,
    I32* bid, I32* rel, I32* error
) {
    const std::size_t total = std::size_t(n) * dims;
    const float scale = __fmul_rn(2.0f, eb);
    for (std::size_t i = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i < total; i += std::size_t(blockDim.x)*gridDim.x) {
        const int d = static_cast<int>(i / n);
        const float value = points[i];
        const float quotient = __fdiv_rn(__fsub_rn(value, minima[d]), scale);
        if (!isfinite(value) || !isfinite(quotient) || quotient < 0.0f || quotient >= 0x1p31f) {
            atomicOr(error, 1);
            bid[i] = 0; rel[i] = 0;
            continue;
        }
        const I32 q = __float2int_rd(quotient);
        const I32 block = q / p;
        bid[i] = block;
        rel[i] = q - block*p;
    }
}
__device__ inline std::uint32_t mix32(std::uint32_t v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return v;
}
__device__ inline bool same_block(const I32* bid, int n, int dims, I32 a, I32 b) {
    for (int d = 0; d < dims; ++d)
        if (bid[std::size_t(d)*n+a] != bid[std::size_t(d)*n+b]) return false;
    return true;
}
__global__ void build_hash_block_code_kernel(
    const I32* bid, int n, int dims, std::uint32_t capacity, I32* representatives,
    std::uint32_t* codes, I32* ids, I32* error
) {
    for (std::size_t index = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         index < std::size_t(n); index += std::size_t(blockDim.x)*gridDim.x) {
        const I32 i = static_cast<I32>(index);
        ids[i] = i;
        std::uint32_t hash = 0x9e3779b9u;
        for (int d = 0; d < dims; ++d) {
            const auto value = static_cast<std::uint32_t>(bid[std::size_t(d)*n+i]);
            hash ^= mix32(value + 0x9e3779b9u + (std::uint32_t(d)<<16) + (hash<<6) + (hash>>2));
        }
        bool found = false;
        // Fixed-capacity linear probing; no mid-pipeline reallocation/retry.
        for (std::uint32_t probe = 0; probe < capacity; ++probe) {
            const auto slot = (hash+probe) & (capacity-1u);
            const I32 old = atomicCAS(representatives+slot, -1, i);
            if (old == -1 || same_block(bid, n, dims, old, i)) {
                codes[i] = slot; found = true; break;
            }
        }
        if (!found) { codes[i] = 0xffffffffu; atomicOr(error, 2); }
    }
}
__global__ void build_next_particle_kernel(
    const std::uint32_t* sorted_codes, const I32* sorted_ids, int n, I32* next
) {
    for (std::size_t i = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i < std::size_t(n); i += std::size_t(blockDim.x)*gridDim.x) {
        next[sorted_ids[i]] = i+1 < std::size_t(n) && sorted_codes[i+1] == sorted_codes[i]
            ? sorted_ids[i+1] : -1;
    }
}
__global__ void make_choice_kernel(
    const I32* bid, const I32* next, int n, int dims, I32 threshold,
    std::uint32_t global_max, I32* idrel, I32* block_flags
) {
    for (std::size_t i = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i < std::size_t(n); i += std::size_t(blockDim.x)*gridDim.x) {
        std::uint32_t local_max = 0;
        for (int d = 0; d < dims; ++d) {
            const auto v = static_cast<std::uint32_t>(bid[std::size_t(d)*n+i]);
            local_max = v > local_max ? v : local_max;
        }
        const I32 distance = next[i] > static_cast<I32>(i) ? next[i]-static_cast<I32>(i) : 0;
        const bool relative = use_relative(distance, local_max, global_max, dims, threshold);
        idrel[i] = relative ? distance : 0;
        block_flags[i] = relative ? 0 : 1;
    }
}
__global__ void gather_records_kernel(
    const I32* indices, Matrix source, const I32* count, I32* output, int output_stride
) {
    for (std::size_t j = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         j < std::size_t(*count); j += std::size_t(blockDim.x)*gridDim.x) {
        const I32 i = indices[j];
        for (int d = 0; d < source.dims; ++d)
            output[std::size_t(d)*output_stride+j] = source.data[std::size_t(d)*source.stride+i];
    }
}
__global__ void split_blocks_kernel(
    Matrix blocks, BlockStats stats, Wrap wrap,
    I32* temporary_diff, I32* temporary_keep, I32* choices, I32* diff_flags, I32* keep_flags
) {
    for (std::size_t i = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i < std::size_t(blocks.n); i += std::size_t(blockDim.x)*gridDim.x) {
        std::uint32_t local_max = 0, local_difference = 0;
        for (int d = 0; d < blocks.dims; ++d) {
            const auto at = std::size_t(d)*blocks.stride+i;
            const I32 value = blocks.data[at];
            local_max = static_cast<std::uint32_t>(value) > local_max ? value : local_max;
            if (i > 0) {
                const I32 previous = blocks.data[at-1];
                const auto difference = static_cast<std::uint32_t>(
                    previous >= value ? previous-value : value-previous);
                local_difference = difference > local_difference ? difference : local_difference;
            }
        }
        const bool take_diff = i > 0 && use_difference(local_max, local_difference, stats, blocks.dims);
        choices[i] = take_diff ? 1 : 0;
        diff_flags[i] = take_diff ? 1 : 0;
        keep_flags[i] = take_diff ? 0 : 1;
        for (int d = 0; d < blocks.dims; ++d) {
            const auto at = std::size_t(d)*blocks.stride+i;
            const I32 value = blocks.data[at];
            temporary_diff[at] = take_diff ? wrapped_symbol(blocks.data[at-1], value, wrap) : -1;
            temporary_keep[at] = take_diff ? -1 : value;
        }
    }
}

struct FixedDimension { I32 minimum = 0; int width = 1; std::size_t bytes = 0; };
__global__ void prepare_tags_and_raw_kernel(
    Matrix input, unsigned char* tags, U64* bits, unsigned char* raw
) {
    for (std::size_t i = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i < std::size_t(input.n); i += std::size_t(blockDim.x)*gridDim.x) {
        int width = 1;
        for (int d = 0; d < input.dims; ++d) {
            const auto value = static_cast<std::uint32_t>(input.data[std::size_t(d)*input.stride+i]);
            const int w = bit_width(value);
            width = w > width ? w : width;
            const auto offset = (i*input.dims+d)*4;
            for (int b = 0; b < 4; ++b)
                raw[offset+b] = static_cast<unsigned char>(value >> (8*b));
        }
        tags[i] = static_cast<unsigned char>(width);
        bits[i] = U64(width)*input.dims;
    }
}
__global__ void pack_fixed_kernel(
    const I32* values, int n, I32 minimum, int width, unsigned char* output, std::size_t bytes
) {
    const std::size_t valid_bits = std::size_t(n)*width;
    for (std::size_t byte = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         byte < bytes; byte += std::size_t(blockDim.x)*gridDim.x) {
        unsigned char packed = 0;
        for (int b = 0; b < 8; ++b) {
            const auto position = byte*8+b;
            unsigned value_bit = 0;
            if (position < valid_bits) {
                const auto i = position / width;
                const auto offset = static_cast<std::uint32_t>(values[i])
                                  - static_cast<std::uint32_t>(minimum);
                value_bit = (offset >> (width-1-static_cast<int>(position%width))) & 1u;
            }
            packed = static_cast<unsigned char>((packed<<1) | value_bit);
        }
        output[byte] = packed;
    }
}
__global__ void save_total_bits_kernel(const U64* offsets, const U64* bits, int n, U64* total) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *total = n ? offsets[n-1]+bits[n-1] : 0;
}
__global__ void pack_tagged_kernel(
    Matrix input, const unsigned char* tags, const U64* offsets, std::uint32_t* words
) {
    for (std::size_t i = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i < std::size_t(input.n); i += std::size_t(blockDim.x)*gridDim.x) {
        const int width = tags[i];
        for (int d = 0; d < input.dims; ++d) {
            const auto value = static_cast<std::uint32_t>(input.data[std::size_t(d)*input.stride+i]);
            const U64 begin = offsets[i]+U64(d)*width;
            for (int bit = 0; bit < width; ++bit) {
                if ((value >> (width-1-bit)) & 1u) {
                    const U64 pos = begin+bit;
                    const std::size_t byte = static_cast<std::size_t>(pos/8);
                    const int shift = int(byte%4)*8+7-int(pos%8);
                    atomicOr(words+byte/4, std::uint32_t(1)<<shift);
                }
            }
        }
    }
}


enum class Codec : unsigned char { Ans = 1, Zstd = 2 };
enum class Encoding : unsigned char { Empty = 0, Tagged = 1, Fixed = 2, Raw = 3 };

struct BatchJob {
    std::size_t raw_bytes = 0, first_chunk = 0, chunks = 0;
};

// All payload/workspace allocations happen in this constructor, before H2D.
// add/upload/launch operate on the actual chunk count; no dummy codec jobs.
class CodecBatch {
public:
    Codec codec;
    std::size_t chunk_bytes, maximum_chunk, output_limit = 0, output_stride = 0;
    std::size_t maximum_chunks = 0, maximum_input = 0, temporary_bytes = 0;
    std::vector<std::size_t> capacities;
    std::vector<BatchJob> jobs;
    std::vector<const void*> inputs;
    std::vector<void*> outputs;
    std::vector<std::size_t> input_bytes, compressed_bytes;
    std::vector<nvcompStatus_t> statuses;
    DeviceBuffer<void*> d_inputs, d_outputs;
    DeviceBuffer<std::size_t> d_input_bytes, d_compressed_bytes;
    DeviceBuffer<nvcompStatus_t> d_statuses;
    DeviceBuffer<unsigned char> d_temporary, d_storage;

    nvcompBatchedANSCompressOpts_t ans_opts = nvcompBatchedANSCompressDefaultOpts;
    nvcompBatchedZstdCompressOpts_t zstd_opts = nvcompBatchedZstdCompressDefaultOpts;

    CodecBatch(Codec which, std::vector<std::size_t> upper_inputs)
        : codec(which), chunk_bytes(which == Codec::Ans ? kAnsChunkBytes : kZstdChunkBytes),
          maximum_chunk(0), capacities(std::move(upper_inputs)) {
        ans_opts.type = nvcomp_rANS;
        ans_opts.data_type = NVCOMP_TYPE_UCHAR;
        for (const auto bytes : capacities) {
            maximum_chunks = checked_add(maximum_chunks, ceil_div(bytes, chunk_bytes));
            maximum_input = checked_add(maximum_input, bytes);
            maximum_chunk = std::max(maximum_chunk, std::min(bytes, chunk_bytes));
        }
        require(maximum_chunks > 0 && maximum_chunks <= std::numeric_limits<std::uint32_t>::max(),
                "too many codec chunks");
        nvcompAlignmentRequirements_t alignment{};
        if (codec == Codec::Ans) {
            LCPS_NVCOMP(nvcompBatchedANSCompressGetRequiredAlignments(ans_opts, &alignment));
            LCPS_NVCOMP(nvcompBatchedANSCompressGetTempSizeAsync(
                maximum_chunks, maximum_chunk, ans_opts, &temporary_bytes, maximum_input));
            LCPS_NVCOMP(nvcompBatchedANSCompressGetMaxOutputChunkSize(
                maximum_chunk, ans_opts, &output_limit));
        } else {
            LCPS_NVCOMP(nvcompBatchedZstdCompressGetRequiredAlignments(zstd_opts, &alignment));
            LCPS_NVCOMP(nvcompBatchedZstdCompressGetTempSizeAsync(
                maximum_chunks, maximum_chunk, zstd_opts, &temporary_bytes, maximum_input));
            LCPS_NVCOMP(nvcompBatchedZstdCompressGetMaxOutputChunkSize(
                maximum_chunk, zstd_opts, &output_limit));
        }
        // cudaMalloc bases are >=256-byte aligned. Chunk starts and output
        // strides are explicitly aligned too (a raw max-size bound need not be).
        require(alignment.input > 0 && alignment.output > 0 && alignment.temp > 0
                && 256 % alignment.input == 0 && 256 % alignment.output == 0
                && 256 % alignment.temp == 0 && chunk_bytes % alignment.input == 0,
                "unsupported nvCOMP buffer alignment");
        output_stride = align_up(output_limit, 256);
        d_inputs.allocate(maximum_chunks);
        d_outputs.allocate(maximum_chunks);
        d_input_bytes.allocate(maximum_chunks);
        d_compressed_bytes.allocate(maximum_chunks);
        d_statuses.allocate(maximum_chunks);
        d_temporary.allocate(std::max<std::size_t>(1, temporary_bytes));
        d_storage.allocate(checked_mul(maximum_chunks, output_stride));
        jobs.reserve(capacities.size());
        inputs.reserve(maximum_chunks); outputs.reserve(maximum_chunks);
        input_bytes.reserve(maximum_chunks);
        compressed_bytes.resize(maximum_chunks);
        statuses.resize(maximum_chunks);
    }

    int add(const unsigned char* data, std::size_t bytes) {
        const auto id = jobs.size();
        require(id < capacities.size() && bytes <= capacities[id], "codec input exceeds reservation");
        const std::size_t chunks = ceil_div(bytes, chunk_bytes);
        BatchJob job{bytes, inputs.size(), chunks};
        for (std::size_t i = 0; i < chunks; ++i) {
            const auto offset = i*chunk_bytes;
            const auto index = inputs.size();
            require(index < maximum_chunks, "codec chunk reservation exceeded");
            inputs.push_back(data+offset);
            input_bytes.push_back(std::min(chunk_bytes, bytes-offset));
            outputs.push_back(d_storage.ptr+index*output_stride);
        }
        jobs.push_back(job);
        return static_cast<int>(id);
    }
    void upload_descriptors(cudaStream_t stream) {
        const auto n = inputs.size();
        if (!n) return;
        LCPS_CUDA(cudaMemcpyAsync(d_inputs.ptr, inputs.data(), n*sizeof(void*), cudaMemcpyHostToDevice, stream));
        LCPS_CUDA(cudaMemcpyAsync(d_outputs.ptr, outputs.data(), n*sizeof(void*), cudaMemcpyHostToDevice, stream));
        LCPS_CUDA(cudaMemcpyAsync(d_input_bytes.ptr, input_bytes.data(), n*sizeof(std::size_t),
                                  cudaMemcpyHostToDevice, stream));
    }
    void launch(cudaStream_t stream) {
        const auto n = inputs.size();
        if (!n) return;
        std::size_t needed = 0;
        if (codec == Codec::Ans) {
            LCPS_NVCOMP(nvcompBatchedANSCompressGetTempSizeAsync(
                n, maximum_chunk, ans_opts, &needed, maximum_input));
        } else {
            LCPS_NVCOMP(nvcompBatchedZstdCompressGetTempSizeAsync(
                n, maximum_chunk, zstd_opts, &needed, maximum_input));
        }
        require(needed <= d_temporary.count, "nvCOMP workspace exceeds precomputed reservation");
        if (codec == Codec::Ans) {
            LCPS_NVCOMP(nvcompBatchedANSCompressAsync(
                reinterpret_cast<const void* const*>(d_inputs.ptr), d_input_bytes.ptr,
                maximum_chunk, n, d_temporary.ptr, d_temporary.count, d_outputs.ptr,
                d_compressed_bytes.ptr, ans_opts, d_statuses.ptr, stream));
        } else {
            LCPS_NVCOMP(nvcompBatchedZstdCompressAsync(
                reinterpret_cast<const void* const*>(d_inputs.ptr), d_input_bytes.ptr,
                maximum_chunk, n, d_temporary.ptr, d_temporary.count,
                reinterpret_cast<void* const*>(d_outputs.ptr), d_compressed_bytes.ptr,
                zstd_opts, d_statuses.ptr, stream));
        }
    }
    void download_sizes(cudaStream_t stream) {
        const auto n = inputs.size();
        if (!n) return;
        LCPS_CUDA(cudaMemcpyAsync(compressed_bytes.data(), d_compressed_bytes.ptr,
                                  n*sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
        LCPS_CUDA(cudaMemcpyAsync(statuses.data(), d_statuses.ptr,
                                  n*sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, stream));
    }
    void validate_results() const {
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (statuses[i] != nvcompSuccess) {
                throw std::runtime_error(std::string(codec == Codec::Ans ? "ANS" : "Zstd")
                    + " chunk " + std::to_string(i) + " failed, status "
                    + std::to_string(static_cast<int>(statuses[i])));
            }
            require(compressed_bytes[i] > 0 && compressed_bytes[i] <= output_limit,
                    "invalid nvCOMP compressed size");
        }
    }
};

struct CandidateInputs {
    Matrix matrix;
    DeviceBuffer<unsigned char> fixed, tagged, tags, raw;
    std::vector<FixedDimension> dimensions;
    std::size_t fixed_bytes = 0, tagged_bytes = 0, raw_bytes = 0;
    U64 tagged_bits = 0;
    int tag_job = -1, raw_job = -1, fixed_job = -1, values_job = -1;

    void allocate(int dims, int n_capacity) {
        matrix.dims = dims; matrix.stride = n_capacity;
        const auto capacity = checked_mul(checked_mul(std::size_t(dims), n_capacity), 4);
        fixed.allocate(capacity);
        tagged.allocate(align_up(capacity, 4));
        tags.allocate(n_capacity);
        raw.allocate(capacity);
        dimensions.resize(dims);
    }
};

struct Section { const CodecBatch* batch = nullptr; int job = -1; };
struct Component {
    Encoding encoding = Encoding::Empty;
    const CandidateInputs* input = nullptr;
    std::array<Section, 2> sections{};
    int section_count = 0;

    std::size_t wire_size() const {
        std::size_t bytes = 14; // encoding + D + count + section_count
        if (encoding == Encoding::Fixed) bytes = checked_add(bytes, checked_mul(input->matrix.dims, 13));
        if (encoding == Encoding::Tagged) bytes = checked_add(bytes, 8);
        for (int s = 0; s < section_count; ++s) {
            const auto& batch = *sections[s].batch;
            const auto& job = batch.jobs.at(sections[s].job);
            bytes = checked_add(bytes, checked_add(13, checked_mul(job.chunks, 16)));
            for (std::size_t i = 0; i < job.chunks; ++i)
                bytes = checked_add(bytes, batch.compressed_bytes[job.first_chunk+i]);
        }
        return bytes;
    }
};

static Component select_component(const CandidateInputs& input, const CodecBatch& ans, const CodecBatch& zstd) {
    Component best;
    best.input = &input;
    if (input.matrix.n == 0) return best;
    best.encoding = Encoding::Tagged;
    best.sections[0] = {&ans, input.tag_job};
    best.sections[1] = {&zstd, input.values_job};
    best.section_count = 2;
    Component fixed;
    fixed.encoding = Encoding::Fixed; fixed.input = &input;
    fixed.sections[0] = {&zstd, input.fixed_job}; fixed.section_count = 1;
    if (fixed.wire_size() < best.wire_size()) best = fixed;
    Component raw;
    raw.encoding = Encoding::Raw; raw.input = &input;
    raw.sections[0] = {&ans, input.raw_job}; raw.section_count = 1;
    if (raw.wire_size() < best.wire_size()) best = raw;
    return best;
}

struct GatherCopy {
    const unsigned char* source;
    std::size_t destination, bytes;
};
__global__ void gather_encoded_chunks_kernel(
    const GatherCopy* copies, std::size_t count, unsigned char* output
) {
    for (std::size_t c = blockIdx.x; c < count; c += gridDim.x) {
        const auto copy = copies[c];
        for (std::size_t i = threadIdx.x; i < copy.bytes; i += blockDim.x)
            output[copy.destination+i] = copy.source[i];
    }
}

class ByteWriter {
    unsigned char* output_;
    std::size_t capacity_, position_ = 0;
public:
    ByteWriter(unsigned char* output, std::size_t capacity) : output_(output), capacity_(capacity) {}
    void raw(const void* source, std::size_t bytes) {
        require(bytes <= capacity_-position_, "LCG2 serialization overflow");
        if (bytes) std::memcpy(output_+position_, source, bytes);
        position_ += bytes;
    }
    void u8(std::uint8_t value) { raw(&value, 1); }
    void u32(std::uint32_t value) { for (int i = 0; i < 4; ++i) u8(static_cast<std::uint8_t>(value >> (8*i))); }
    void u64(std::uint64_t value) { for (int i = 0; i < 8; ++i) u8(static_cast<std::uint8_t>(value >> (8*i))); }
    void i32(I32 value) { u32(static_cast<std::uint32_t>(value)); }
    void f32(float value) { std::uint32_t bits; std::memcpy(&bits, &value, 4); u32(bits); }
    std::size_t size() const { return position_; }
};

static void write_component(
    ByteWriter& writer, const Component& component, const unsigned char* staging, std::size_t& staging_offset
) {
    const auto& input = *component.input;
    writer.u8(static_cast<unsigned char>(component.encoding));
    writer.u32(input.matrix.dims);
    writer.u64(input.matrix.n);
    if (component.encoding == Encoding::Fixed) {
        for (const auto& dimension : input.dimensions) {
            writer.i32(dimension.minimum); writer.u8(dimension.width); writer.u64(dimension.bytes);
        }
    } else if (component.encoding == Encoding::Tagged) {
        writer.u64(input.tagged_bits);
    }
    writer.u8(component.section_count);
    for (int s = 0; s < component.section_count; ++s) {
        const auto& batch = *component.sections[s].batch;
        const auto& job = batch.jobs.at(component.sections[s].job);
        writer.u8(static_cast<unsigned char>(batch.codec));
        writer.u64(job.raw_bytes);
        writer.u32(static_cast<std::uint32_t>(job.chunks));
        std::size_t payload = 0;
        for (std::size_t i = 0; i < job.chunks; ++i) {
            const auto index = job.first_chunk+i;
            writer.u64(batch.input_bytes[index]); writer.u64(batch.compressed_bytes[index]);
            payload = checked_add(payload, batch.compressed_bytes[index]);
        }
        writer.raw(staging+staging_offset, payload);
        staging_offset = checked_add(staging_offset, payload);
    }
}


static py::bytes lcps_compress2_gpu_cpp(
    py::object points_obj, float eb, I32 p, I32 threshold, I32 diff
) {
    (void)diff; // Legacy API argument; CPU currently uses the fixed saving threshold 0.
    require(eb > 0.0f && std::isfinite(eb) && std::isfinite(2.0f*eb), "eb must be positive and finite");
    require(p > 0, "p must be positive");
    auto points = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(points_obj);
    require(bool(points) && points.ndim() == 2, "points must be a float32 SoA array of shape (D, N)");
    require(points.shape(0) > 0 && points.shape(1) > 0
            && points.shape(0) <= std::numeric_limits<I32>::max()
            && points.shape(1) <= std::numeric_limits<I32>::max(), "invalid int32 array shape");
    const int dims = static_cast<int>(points.shape(0));
    const int n = static_cast<int>(points.shape(1));
    const std::size_t elements = checked_mul(std::size_t(n), dims);
    const cudaStream_t stream = 0;
    std::uint64_t hash_capacity64 = 1;
    const auto requested_hash_capacity = checked_mul(std::size_t(n), 16);
    while (hash_capacity64 < requested_hash_capacity) hash_capacity64 *= 2;
    require(hash_capacity64 <= std::numeric_limits<std::uint32_t>::max(), "hash table exceeds uint32 indexing");
    const auto hash_capacity = static_cast<std::uint32_t>(hash_capacity64);

    // Phase 0: query upper bounds and allocate ALL large buffers. No array
    // transfer, kernel launch, memset, or data-dependent allocation precedes it.
    std::size_t sort_bytes = 0, select_bytes = 0, scan_bytes = 0;
    std::size_t float_reduce_bytes = 0, block_reduce_bytes = 0, minmax_reduce_bytes = 0;
    LCPS_CUDA(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, static_cast<std::uint32_t*>(nullptr),
        static_cast<std::uint32_t*>(nullptr), static_cast<I32*>(nullptr),
        static_cast<I32*>(nullptr), n, 0, 32, stream));
    LCPS_CUDA(cub::DeviceSelect::Flagged(
        nullptr, select_bytes, CountIterator(0), static_cast<I32*>(nullptr),
        static_cast<I32*>(nullptr), static_cast<I32*>(nullptr), n, stream));
    LCPS_CUDA(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, static_cast<U64*>(nullptr), static_cast<U64*>(nullptr), n, stream));
    LCPS_CUDA(cub::DeviceReduce::Min(
        nullptr, float_reduce_bytes, static_cast<const float*>(nullptr),
        static_cast<float*>(nullptr), n, stream));
    const BlockStats identity{std::numeric_limits<I32>::max(), 0, 0};
    const BlockStatsIterator query_blocks(CountIterator(0), ReadBlockStats{Matrix{nullptr, dims, n, n}, true});
    LCPS_CUDA(cub::DeviceReduce::Reduce(
        nullptr, block_reduce_bytes, query_blocks, static_cast<BlockStats*>(nullptr), n,
        MergeBlockStats{}, identity, stream));
    const MinMaxIterator query_minmax(static_cast<const I32*>(nullptr), ReadMinMax{});
    LCPS_CUDA(cub::DeviceReduce::Reduce(
        nullptr, minmax_reduce_bytes, query_minmax, static_cast<MinMax*>(nullptr), n,
        MergeMinMax{}, MinMax{std::numeric_limits<I32>::max(), 0}, stream));
    const auto scratch_bytes = std::max({std::size_t(1), sort_bytes, select_bytes, scan_bytes,
        float_reduce_bytes, block_reduce_bytes, minmax_reduce_bytes});

    DeviceBuffer<float> d_points(elements), d_minima(dims);
    DeviceBuffer<I32> d_bid(elements), d_rel(elements), d_blocks(elements);
    DeviceBuffer<I32> d_temporary_diff(elements), d_temporary_keep(elements);
    DeviceBuffer<I32> d_diff(elements), d_keep(elements);
    DeviceBuffer<I32> d_idrel(n), d_choice2(n), d_next(n);
    DeviceBuffer<I32> d_flags(n), d_diff_flags(n), d_keep_flags(n), d_selected(n);
    DeviceBuffer<I32> d_ids_in(n), d_ids_out(n), d_representatives(hash_capacity);
    DeviceBuffer<std::uint32_t> d_codes_in(n), d_codes_out(n);
    DeviceBuffer<I32> d_counts(3), d_error(1);
    DeviceBuffer<BlockStats> d_block_stats(1);
    DeviceBuffer<U64> d_bits(n), d_offsets(n), d_total_bits(kStreamCount);
    DeviceBuffer<unsigned char> d_scratch(scratch_bytes);
    const std::size_t stat_count = checked_add(checked_mul(dims, 3), 2);
    DeviceBuffer<MinMax> d_minmax(stat_count);

    std::array<CandidateInputs, kStreamCount> candidates;
    const std::array<int, kStreamCount> component_dims{{1, dims, dims, 1, dims}};
    std::vector<std::size_t> ans_capacities, zstd_capacities;
    ans_capacities.reserve(2*kStreamCount); zstd_capacities.reserve(2*kStreamCount);
    for (int s = 0; s < kStreamCount; ++s) {
        candidates[s].allocate(component_dims[s], n);
        ans_capacities.push_back(n);
        ans_capacities.push_back(candidates[s].raw.count);
        zstd_capacities.push_back(candidates[s].fixed.count);
        zstd_capacities.push_back(candidates[s].tagged.count);
    }
    CodecBatch ans(Codec::Ans, std::move(ans_capacities));
    CodecBatch zstd(Codec::Zstd, std::move(zstd_capacities));

    // Every winner's complete size is <= its Fixed candidate. Therefore this
    // bound safely holds all selected payloads, without reserving space for
    // every losing candidate a second time.
    std::size_t final_capacity = 0;
    for (const auto& input : candidates) {
        const auto chunks = ceil_div(input.fixed.count, kZstdChunkBytes);
        auto bound = checked_add(27, checked_mul(input.matrix.dims, 13));
        bound = checked_add(bound, checked_mul(chunks, 16));
        bound = checked_add(bound, checked_mul(chunks, zstd.output_limit));
        final_capacity = checked_add(final_capacity, bound);
    }
    DeviceBuffer<unsigned char> d_final_payload(final_capacity);
    const auto gather_capacity = checked_add(ans.maximum_chunks, zstd.maximum_chunks);
    DeviceBuffer<GatherCopy> d_gather_copies(gather_capacity);
    HostStaging host_payload(final_capacity);
    std::vector<GatherCopy> gather_copies;
    gather_copies.reserve(gather_capacity);
    std::vector<float> minima(dims);
    std::vector<MinMax> minmax(stat_count);
    std::array<U64, kStreamCount> total_bits{};
    std::array<I32, 3> counts{};
    I32 error = 0;
    BlockStats stats{};

    // Phase 1: the only bulk H2D transfer is the original float32 coordinate array.
    LCPS_CUDA(cudaMemcpyAsync(d_points.ptr, points.data(), checked_mul(elements, sizeof(float)),
                              cudaMemcpyHostToDevice, stream));
    LCPS_CUDA(cudaStreamSynchronize(stream));
    LCPS_CUDA(cudaMemsetAsync(d_error.ptr, 0, sizeof(I32), stream));
    const auto t0 = std::chrono::steady_clock::now();
    for (int d = 0; d < dims; ++d) {
        auto available = d_scratch.count;
        LCPS_CUDA(cub::DeviceReduce::Min(
            d_scratch.ptr, available, d_points.ptr+std::size_t(d)*n, d_minima.ptr+d, n, stream));
    }
    quantize_bid_rel_kernel<<<grid_for(elements), kThreads, 0, stream>>>(
        d_points.ptr, d_minima.ptr, n, dims, eb, p, d_bid.ptr, d_rel.ptr, d_error.ptr);
    LCPS_CUDA(cudaGetLastError());
    {
        auto available = d_scratch.count;
        const BlockStatsIterator input(CountIterator(0), ReadBlockStats{Matrix{d_bid.ptr, dims, n, n}, false});
        LCPS_CUDA(cub::DeviceReduce::Reduce(
            d_scratch.ptr, available, input, d_block_stats.ptr, n, MergeBlockStats{}, identity, stream));
    }
    const auto t1 = std::chrono::steady_clock::now();
    // Small metadata boundary A: minima, one global block statistic, one status.
    LCPS_CUDA(cudaMemcpyAsync(minima.data(), d_minima.ptr, dims*sizeof(float), cudaMemcpyDeviceToHost, stream));
    LCPS_CUDA(cudaMemcpyAsync(&stats, d_block_stats.ptr, sizeof(stats), cudaMemcpyDeviceToHost, stream));
    LCPS_CUDA(cudaMemcpyAsync(&error, d_error.ptr, sizeof(error), cudaMemcpyDeviceToHost, stream));
    LCPS_CUDA(cudaStreamSynchronize(stream));
    require(error == 0, "non-finite input or float-quantized coordinate outside nonnegative int32");
    const auto t2 = std::chrono::steady_clock::now();
    const auto global_max = static_cast<std::uint32_t>(stats.maximum);
    const auto t3 = std::chrono::steady_clock::now();

    // Phase 2: preserve same-block successor semantics using a stable radix sort.
    const auto t4 = std::chrono::steady_clock::now();
    LCPS_CUDA(cudaMemsetAsync(d_representatives.ptr, 0xff,
                              checked_mul(hash_capacity, sizeof(I32)), stream));
    build_hash_block_code_kernel<<<grid_for(n), kThreads, 0, stream>>>(
        d_bid.ptr, n, dims, hash_capacity, d_representatives.ptr,
        d_codes_in.ptr, d_ids_in.ptr, d_error.ptr);
    LCPS_CUDA(cudaGetLastError());
    {
        auto available = d_scratch.count;
        LCPS_CUDA(cub::DeviceRadixSort::SortPairs(
            d_scratch.ptr, available, d_codes_in.ptr, d_codes_out.ptr, d_ids_in.ptr, d_ids_out.ptr,
            n, 0, 32, stream));
    }
    build_next_particle_kernel<<<grid_for(n), kThreads, 0, stream>>>(
        d_codes_out.ptr, d_ids_out.ptr, n, d_next.ptr);
    LCPS_CUDA(cudaGetLastError());
    make_choice_kernel<<<grid_for(n), kThreads, 0, stream>>>(
        d_bid.ptr, d_next.ptr, n, dims, threshold, global_max, d_idrel.ptr, d_flags.ptr);
    LCPS_CUDA(cudaGetLastError());
    {
        auto available = d_scratch.count;
        LCPS_CUDA(cub::DeviceSelect::Flagged(
            d_scratch.ptr, available, CountIterator(0), d_flags.ptr, d_selected.ptr,
            d_counts.ptr, n, stream));
    }
    gather_records_kernel<<<grid_for(n), kThreads, 0, stream>>>(
        d_selected.ptr, Matrix{d_bid.ptr, dims, n, n}, d_counts.ptr, d_blocks.ptr, n);
    LCPS_CUDA(cudaGetLastError());
    // Small metadata boundary B: compact block count and hash status.
    LCPS_CUDA(cudaMemcpyAsync(counts.data(), d_counts.ptr, sizeof(I32), cudaMemcpyDeviceToHost, stream));
    LCPS_CUDA(cudaMemcpyAsync(&error, d_error.ptr, sizeof(error), cudaMemcpyDeviceToHost, stream));
    LCPS_CUDA(cudaStreamSynchronize(stream));
    require(error == 0, "hash table insertion failed");
    const int block_count = counts[0];
    require(block_count > 0 && block_count <= n, "invalid compact block count");
    const auto t5 = std::chrono::steady_clock::now();

    // Phase 3: reductions and stable, record-wise diff/keep compaction.
    const Matrix compact_blocks{d_blocks.ptr, dims, n, block_count};
    {
        auto available = d_scratch.count;
        const BlockStatsIterator input(CountIterator(0), ReadBlockStats{compact_blocks, true});
        LCPS_CUDA(cub::DeviceReduce::Reduce(
            d_scratch.ptr, available, input, d_block_stats.ptr, block_count,
            MergeBlockStats{}, identity, stream));
    }
    // Small metadata boundary C: one statistic; wrap arithmetic is trivial on CPU.
    LCPS_CUDA(cudaMemcpyAsync(&stats, d_block_stats.ptr, sizeof(stats), cudaMemcpyDeviceToHost, stream));
    LCPS_CUDA(cudaStreamSynchronize(stream));
    const Wrap wrap = make_wrap(stats);
    split_blocks_kernel<<<grid_for(block_count), kThreads, 0, stream>>>(
        compact_blocks, stats, wrap, d_temporary_diff.ptr, d_temporary_keep.ptr,
        d_choice2.ptr, d_diff_flags.ptr, d_keep_flags.ptr);
    LCPS_CUDA(cudaGetLastError());
    {
        auto available = d_scratch.count;
        LCPS_CUDA(cub::DeviceSelect::Flagged(
            d_scratch.ptr, available, CountIterator(0), d_diff_flags.ptr, d_selected.ptr,
            d_counts.ptr+1, block_count, stream));
    }
    gather_records_kernel<<<grid_for(block_count), kThreads, 0, stream>>>(
        d_selected.ptr, Matrix{d_temporary_diff.ptr, dims, n, block_count}, d_counts.ptr+1, d_diff.ptr, n);
    LCPS_CUDA(cudaGetLastError());
    {
        auto available = d_scratch.count;
        LCPS_CUDA(cub::DeviceSelect::Flagged(
            d_scratch.ptr, available, CountIterator(0), d_keep_flags.ptr, d_selected.ptr,
            d_counts.ptr+2, block_count, stream));
    }
    gather_records_kernel<<<grid_for(block_count), kThreads, 0, stream>>>(
        d_selected.ptr, Matrix{d_temporary_keep.ptr, dims, n, block_count}, d_counts.ptr+2, d_keep.ptr, n);
    LCPS_CUDA(cudaGetLastError());
    // Small metadata boundary D: only three record counts, never the integer streams.
    LCPS_CUDA(cudaMemcpyAsync(counts.data(), d_counts.ptr, counts.size()*sizeof(I32),
                              cudaMemcpyDeviceToHost, stream));
    LCPS_CUDA(cudaStreamSynchronize(stream));
    require(counts[1] >= 0 && counts[2] > 0
            && static_cast<std::int64_t>(counts[1])+counts[2] == block_count,
            "invalid diff/keep compaction counts");
    candidates[0].matrix = {d_idrel.ptr, 1, n, n};
    candidates[1].matrix = {d_diff.ptr, dims, n, counts[1]};
    candidates[2].matrix = {d_keep.ptr, dims, n, counts[2]};
    candidates[3].matrix = {d_choice2.ptr, 1, n, block_count};
    candidates[4].matrix = {d_rel.ptr, dims, n, n};
    const auto t6 = std::chrono::steady_clock::now();

    // Phase 4: prepare tag bytes and point-major int32 raw bytes. The N-element
    // scan buffers are reused on the same stream, only after each pack completes.
    LCPS_CUDA(cudaMemsetAsync(d_minmax.ptr, 0, stat_count*sizeof(MinMax), stream));
    LCPS_CUDA(cudaMemsetAsync(d_total_bits.ptr, 0, kStreamCount*sizeof(U64), stream));
    std::size_t stat_offset = 0;
    for (int s = 0; s < kStreamCount; ++s) {
        auto& input = candidates[s];
        const auto matrix = input.matrix;
        input.raw_bytes = checked_mul(checked_mul(std::size_t(matrix.n), matrix.dims), 4);
        if (matrix.n) {
            for (int d = 0; d < matrix.dims; ++d) {
                auto available = d_scratch.count;
                const MinMaxIterator values(matrix.data+std::size_t(d)*matrix.stride, ReadMinMax{});
                LCPS_CUDA(cub::DeviceReduce::Reduce(
                    d_scratch.ptr, available, values, d_minmax.ptr+stat_offset+d, matrix.n,
                    MergeMinMax{}, MinMax{std::numeric_limits<I32>::max(), 0}, stream));
            }
            prepare_tags_and_raw_kernel<<<grid_for(matrix.n), kThreads, 0, stream>>>(
                matrix, input.tags.ptr, d_bits.ptr, input.raw.ptr);
            LCPS_CUDA(cudaGetLastError());
            {
                auto available = d_scratch.count;
                LCPS_CUDA(cub::DeviceScan::ExclusiveSum(
                    d_scratch.ptr, available, d_bits.ptr, d_offsets.ptr, matrix.n, stream));
            }
            save_total_bits_kernel<<<1, 1, 0, stream>>>(
                d_offsets.ptr, d_bits.ptr, matrix.n, d_total_bits.ptr+s);
            LCPS_CUDA(cudaGetLastError());
            // 4 bytes/value is an upper bound for all nonnegative int32 tags.
            LCPS_CUDA(cudaMemsetAsync(input.tagged.ptr, 0, input.raw_bytes, stream));
            pack_tagged_kernel<<<grid_for(matrix.n), kThreads, 0, stream>>>(
                matrix, input.tags.ptr, d_offsets.ptr, reinterpret_cast<std::uint32_t*>(input.tagged.ptr));
            LCPS_CUDA(cudaGetLastError());
        }
        stat_offset += matrix.dims;
    }
    // Small metadata boundary E: per-dimension extrema + five total bit counts.
    LCPS_CUDA(cudaMemcpyAsync(minmax.data(), d_minmax.ptr, stat_count*sizeof(MinMax),
                              cudaMemcpyDeviceToHost, stream));
    LCPS_CUDA(cudaMemcpyAsync(total_bits.data(), d_total_bits.ptr, total_bits.size()*sizeof(U64),
                              cudaMemcpyDeviceToHost, stream));
    LCPS_CUDA(cudaStreamSynchronize(stream));
    stat_offset = 0;
    for (int s = 0; s < kStreamCount; ++s) {
        auto& input = candidates[s];
        const auto matrix = input.matrix;
        input.tagged_bits = total_bits[s];
        require(input.tagged_bits <= std::numeric_limits<std::size_t>::max(), "tagged bit count overflow");
        input.tagged_bytes = ceil_div(static_cast<std::size_t>(input.tagged_bits), 8);
        require(input.tagged_bytes <= input.tagged.count, "tagged payload exceeds reservation");
        input.fixed_bytes = 0;
        for (int d = 0; d < matrix.dims; ++d) {
            const auto mm = minmax[stat_offset+d];
            require(mm.minimum >= 0 && mm.maximum >= mm.minimum, "invalid integer stream after compaction");
            auto& dimension = input.dimensions[d];
            dimension.minimum = mm.minimum;
            dimension.width = std::max(1, bit_width(static_cast<std::uint32_t>(mm.maximum)
                                                   - static_cast<std::uint32_t>(mm.minimum)));
            dimension.bytes = ceil_div(checked_mul(std::size_t(matrix.n), dimension.width), 8);
            if (dimension.bytes) {
                pack_fixed_kernel<<<grid_for(dimension.bytes), kThreads, 0, stream>>>(
                    matrix.data+std::size_t(d)*matrix.stride, matrix.n,
                    dimension.minimum, dimension.width, input.fixed.ptr+input.fixed_bytes, dimension.bytes);
                LCPS_CUDA(cudaGetLastError());
            }
            input.fixed_bytes = checked_add(input.fixed_bytes, dimension.bytes);
        }
        stat_offset += matrix.dims;
        require(input.fixed_bytes <= input.fixed.count, "fixed payload exceeds reservation");
        input.tag_job = ans.add(input.tags.ptr, matrix.n);
        input.raw_job = ans.add(input.raw.ptr, input.raw_bytes);
        input.fixed_job = zstd.add(input.fixed.ptr, input.fixed_bytes);
        input.values_job = zstd.add(input.tagged.ptr, input.tagged_bytes);
    }

    // Phase 5: only small pointer/length descriptors go H2D. Exactly one global
    // ANS call and one global Zstd call process all nonempty candidate sections.
    // ANS bytes never pass through Zstd.
    ans.upload_descriptors(stream);
    zstd.upload_descriptors(stream);
    ans.launch(stream);
    zstd.launch(stream);
    // Small metadata boundary F: chunk sizes/statuses for BOTH complete batches.
    ans.download_sizes(stream);
    zstd.download_sizes(stream);
    LCPS_CUDA(cudaStreamSynchronize(stream));
    ans.validate_results();
    zstd.validate_results();

    // Phase 6: cheap CPU selection includes every header and chunk table byte.
    std::array<Component, kStreamCount> selected;
    std::size_t serialized_bytes = checked_add(40, checked_mul(dims, 4));
    std::size_t payload_bytes = 0;
    for (int s = 0; s < kStreamCount; ++s) {
        selected[s] = select_component(candidates[s], ans, zstd);
        serialized_bytes = checked_add(serialized_bytes, checked_add(12, selected[s].wire_size()));
        for (int section = 0; section < selected[s].section_count; ++section) {
            const auto& batch = *selected[s].sections[section].batch;
            const auto& job = batch.jobs.at(selected[s].sections[section].job);
            for (std::size_t c = 0; c < job.chunks; ++c) {
                const auto index = job.first_chunk+c;
                const auto bytes = batch.compressed_bytes[index];
                gather_copies.push_back({batch.d_storage.ptr+index*batch.output_stride, payload_bytes, bytes});
                payload_bytes = checked_add(payload_bytes, bytes);
            }
        }
    }
    require(payload_bytes <= final_capacity && gather_copies.size() <= gather_capacity,
            "selected payload exceeds reservation");
    require(serialized_bytes <= static_cast<std::size_t>(PY_SSIZE_T_MAX), "bitstream exceeds Python bytes size");
    if (!gather_copies.empty()) {
        LCPS_CUDA(cudaMemcpyAsync(d_gather_copies.ptr, gather_copies.data(),
                                  gather_copies.size()*sizeof(GatherCopy), cudaMemcpyHostToDevice, stream));
        gather_encoded_chunks_kernel<<<static_cast<int>(std::min<std::size_t>(65535, gather_copies.size())),
                                        kThreads, 0, stream>>>(
            d_gather_copies.ptr, gather_copies.size(), d_final_payload.ptr);
        LCPS_CUDA(cudaGetLastError());
    }
    LCPS_CUDA(cudaStreamSynchronize(stream));
    const auto t7 = std::chrono::steady_clock::now();

    // Phase 7: the ONLY bulk D2H copies final selected encoded payloads.
    // No losing payload, original array, flags, tags, or packed input is read back.
    if (payload_bytes) {
        LCPS_CUDA(cudaMemcpyAsync(host_payload.ptr, d_final_payload.ptr, payload_bytes,
                                  cudaMemcpyDeviceToHost, stream));
    }
    LCPS_CUDA(cudaStreamSynchronize(stream));
    const float qb = std::chrono::duration<float>(t1-t0+t3-t2).count();
    const float l1 = std::chrono::duration<float>(t5-t4).count();
    const float l2 = std::chrono::duration<float>(t6-t5).count();
    const float l3 = std::chrono::duration<float>(t7-t6).count();
    std::printf("\t%.6f\n\t%.6f\n\t%.6f\n\t%.6f\n", qb, l1, l2, l3);

    // Allocate the final host/Python object ONLY AFTER encoded data has arrived.
    // All large device buffers remain alive until this function returns.
    PyObject* object = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(serialized_bytes));
    if (!object) throw py::error_already_set();
    py::bytes output = py::reinterpret_steal<py::bytes>(object);
    ByteWriter writer(reinterpret_cast<unsigned char*>(PyBytes_AS_STRING(object)), serialized_bytes);
    writer.raw("LCG2", 4); writer.u32(1); writer.u32(dims); writer.u64(n);
    writer.i32(p); writer.f32(eb);
    writer.i32(wrap.minimum); writer.i32(wrap.maximum);
    for (const float minimum : minima) writer.f32(minimum);
    writer.u32(kStreamCount);
    std::size_t consumed_payload = 0;
    for (int s = 0; s < kStreamCount; ++s) {
        writer.raw(kStreamNames[s], 4); writer.u64(selected[s].wire_size());
        write_component(writer, selected[s], host_payload.ptr, consumed_payload);
    }
    require(writer.size() == serialized_bytes && consumed_payload == payload_bytes,
            "LCG2 serialization size mismatch");
    return output;
}

}  // namespace lcps_gpu

PYBIND11_MODULE(lcps_gpu_cpp, m) {
    m.doc() = "GPU Chain compressor: float32/int32 transforms, global nvCOMP ANS/Zstd batches, LCG2 bytes.";
    m.def("lcps_compress2_gpu", &lcps_gpu::lcps_compress2_gpu_cpp,
          py::arg("points"), py::arg("eb"), py::arg("p"), py::arg("threshold"), py::arg("diff"),
          "Compress float32 SoA (D, N) coordinates to one self-contained LCG2 bytes object. "
          "The legacy diff argument is retained; the block-difference saving threshold is 0 bits. "
          "Requires the dedicated LCG2 decoder; float32 rounding may exceed eb.");
}
