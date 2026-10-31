// GPU decoder for LCG1 (legacy) and LCG2 v1 (current GPU compressor).
// Public Python API stays lcps_decompress_gpu(bytes, output_soa=False).
// LCG2 returns float32; LCG1 retains float64. One device arena, one global
// batch per codec, one bulk H2D/D2H, and one final stream synchronization.
// LCG2 integer reconstruction is lossless. Its encoder uses float32
// quantization without coordinate corrections, so a strict original-data eb
// guarantee cannot be recovered from an existing LCG2 v1 bitstream alone.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#if __has_include(<nvcomp/ans.h>)
#include <nvcomp/ans.h>
#elif __has_include(<ans.h>)
#include <ans.h>
#else
#error "nvCOMP ANS header not found. Install nvCOMP and add its include directory."
#endif
#if __has_include(<nvcomp/zstd.h>)
#include <nvcomp/zstd.h>
#elif __has_include(<zstd.h>)
#include <zstd.h>
#else
#error "nvCOMP Zstd header not found. Install nvCOMP with Zstd support and add its include directory."
#endif
#if __has_include(<nvcomp/nvcomp.h>)
#include <nvcomp/nvcomp.h>
#elif __has_include(<nvcomp.h>)
#include <nvcomp.h>
#endif
#include <cub/cub.cuh>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__FAST_MATH__) || defined(__CUDA_FAST_MATH__)
#error "LCG2 reconstruction requires compilation without fast-math"
#endif
#if defined(__CUDA_FTZ) && __CUDA_FTZ
#error "LCG2 reconstruction requires --ftz=false"
#endif

namespace py = pybind11;

#define CUDA_CHECK(stmt) do {                                      \
    cudaError_t _err = (stmt);                                     \
    if (_err != cudaSuccess) {                                     \
        throw std::runtime_error(std::string("CUDA error: ") +     \
            cudaGetErrorString(_err) + " at " + __FILE__ + ":" +  \
            std::to_string(__LINE__));                             \
    }                                                              \
} while (0)

static inline std::string nvcomp_status_string_local(nvcompStatus_t st) {
    switch (st) {
        case nvcompSuccess: return "nvcompSuccess";
#ifdef nvcompErrorInvalidValue
        case nvcompErrorInvalidValue: return "nvcompErrorInvalidValue";
#endif
#ifdef nvcompErrorNotSupported
        case nvcompErrorNotSupported: return "nvcompErrorNotSupported";
#endif
#ifdef nvcompErrorCannotDecompress
        case nvcompErrorCannotDecompress: return "nvcompErrorCannotDecompress";
#endif
#ifdef nvcompErrorOutputBufferTooSmall
        case nvcompErrorOutputBufferTooSmall: return "nvcompErrorOutputBufferTooSmall";
#endif
#ifdef nvcompErrorCudaError
        case nvcompErrorCudaError: return "nvcompErrorCudaError";
#endif
#ifdef nvcompErrorInternal
        case nvcompErrorInternal: return "nvcompErrorInternal";
#endif
        default: return std::string("nvcompStatus(") + std::to_string(static_cast<int>(st)) + ")";
    }
}

#define NVCOMP_CHECK(stmt) do {                                    \
    nvcompStatus_t _st = (stmt);                                   \
    if (_st != nvcompSuccess) {                                    \
        throw std::runtime_error(std::string("nvCOMP error: ") +   \
            nvcomp_status_string_local(_st) + " at " + __FILE__ + ":" +\
            std::to_string(__LINE__));                             \
    }                                                              \
} while (0)

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t count = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { allocate(n); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept { ptr=o.ptr; count=o.count; o.ptr=nullptr; o.count=0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { release(); ptr=o.ptr; count=o.count; o.ptr=nullptr; o.count=0; }
        return *this;
    }
    ~DeviceBuffer() { release(); }
    void allocate(std::size_t n) {
        release(); count = n;
        if (n > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
    }
    void release() { if (ptr) { cudaFree(ptr); ptr=nullptr; } count=0; }
};

template <typename T>
struct PinnedHostBuffer {
    T* ptr = nullptr;
    std::size_t count = 0;
    PinnedHostBuffer() = default;
    explicit PinnedHostBuffer(std::size_t n) { allocate(n); }
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    ~PinnedHostBuffer() { release(); }
    void allocate(std::size_t n) {
        release();
        count = n;
        if (n > 0) {
            CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&ptr), n * sizeof(T)
            ));
        }
    }
    void release() {
        if (ptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
        count = 0;
    }
};

static inline bool host_little_endian() {
    std::uint16_t x = 1;
    return *reinterpret_cast<unsigned char*>(&x) == 1;
}

class Reader {
public:
    const unsigned char* data = nullptr;
    std::size_t size = 0;
    std::size_t pos = 0;
    Reader(const unsigned char* p, std::size_t n) : data(p), size(n), pos(0) {}
    std::size_t remaining() const { return size - pos; }
    void require(std::size_t n) const { if (n > remaining()) throw std::runtime_error("truncated stream"); }
    std::string peek_magic() const {
        if (remaining() < 4) return {};
        return std::string(reinterpret_cast<const char*>(data + pos), 4);
    }
    std::string read_magic() {
        require(4);
        std::string s(reinterpret_cast<const char*>(data + pos), 4);
        pos += 4;
        return s;
    }
    void expect_magic(const char m[4]) {
        std::string s = read_magic();
        if (s != std::string(m, 4)) throw std::runtime_error("bad magic: expected " + std::string(m,4) + ", got " + s);
    }
    template <typename T>
    T read_le() {
        static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "scalar");
        require(sizeof(T));
        unsigned char buf[sizeof(T)];
        std::memcpy(buf, data + pos, sizeof(T));
        pos += sizeof(T);
        if (!host_little_endian()) std::reverse(buf, buf + sizeof(T));
        T v;
        std::memcpy(&v, buf, sizeof(T));
        return v;
    }
    std::uint8_t read_u8() { require(1); return data[pos++]; }
    void skip(std::size_t n) { require(n); pos += n; }
};

static std::size_t u64_to_size(std::uint64_t v, const char* name) {
    if (v > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::runtime_error(std::string(name) + " too large");
    return static_cast<std::size_t>(v);
}
static int u64_to_int(std::uint64_t v, const char* name) {
    if (v > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) throw std::runtime_error(std::string(name) + " too large");
    return static_cast<int>(v);
}

static inline std::size_t align_up_size(std::size_t x, std::size_t a) {
    return (x + a - 1) / a * a;
}

static constexpr std::size_t NVCOMP_CHUNK_PTR_ALIGNMENT = 256;
static constexpr std::size_t UNPLANNED_DEVICE_OFFSET =
    std::numeric_limits<std::size_t>::max();

struct ChunkedCodecSpec {
    std::size_t off = 0;       // offset of the NVA1/ZSTG magic in the container
    std::size_t len = 0;       // complete wrapper length, including payload bytes
    std::size_t data_off = 0;  // first byte of the byte-tight compressed payload
    std::size_t total_uncomp = 0;
    std::size_t max_chunk = 0;
    std::uint32_t num_chunks = 0;
    std::size_t comp_total = 0;
    std::size_t comp_total_aligned = 0;
    std::vector<std::size_t> uncomp_chunk_bytes;
    std::vector<std::size_t> comp_chunk_bytes;
    // Offset of this codec's fully decoded bytes in its global output arena.
    std::size_t decoded_offset = UNPLANNED_DEVICE_OFFSET;
};

struct Nva1Spec : ChunkedCodecSpec {};
struct ZstdSpec : ChunkedCodecSpec {};

// New compressor format: ANS-compressed chunks are concatenated, compressed
// once more by Zstd, and preceded by the original ANS chunk table.
struct AnsZstdSpec {
    std::size_t metadata_off = 0; // total_uncomp field (there is no NVA1 magic)
    std::size_t len = 0;
    std::size_t total_uncomp = 0;
    std::size_t max_chunk = 0;
    std::uint32_t num_chunks = 0;
    std::size_t comp_total = 0;
    std::size_t comp_total_aligned = 0;
    std::vector<std::size_t> uncomp_chunk_bytes;
    std::vector<std::size_t> comp_chunk_bytes;
    // Offset of the final ANS-decoded bytes in the global ANS output arena.
    std::size_t decoded_offset = UNPLANNED_DEVICE_OFFSET;
    ZstdSpec zstd;
};

struct BansSpec {
    std::size_t off = 0;
    std::size_t len = 0;
    int byte_width = 0;
    int n = 0;
    Nva1Spec nva;
};

struct UansSpec {
    std::size_t off = 0;
    std::size_t len = 0;
    int n = 0;
    Nva1Spec nva;
};

struct FixedStreamSpec {
    std::size_t off = 0;
    std::size_t len = 0;
    std::int64_t mn = 0;
    int n = 0;
    int width = 0;
    std::size_t payload_off = 0;
    std::size_t payload_bytes = 0;
};

struct FixedFieldSpec {
    std::int64_t mn = 0;
    int n = 0;
    int width = 0;
    std::size_t payload_offset = 0; // offset in the Zstd-uncompressed buffer
    std::size_t payload_bytes = 0;
};

struct ZFixedStreamSpec {
    std::size_t off = 0;
    std::size_t len = 0;
    FixedFieldSpec field;
    ZstdSpec zstd;
};

struct ZFixedMatrixSpec {
    std::size_t off = 0;
    std::size_t len = 0;
    int D = 0;
    int n = 0;
    std::vector<FixedFieldSpec> fields;
    std::size_t raw_payload_bytes = 0;
    ZstdSpec zstd;
};

struct ZBansSpec {
    std::size_t off = 0;
    std::size_t len = 0;
    int byte_width = 0;
    int n = 0;
    std::size_t payload_bytes = 0;
    AnsZstdSpec ans;
};

struct ZAnsMatrixSpec {
    std::size_t off = 0;
    std::size_t len = 0;
    int D = 0;
    int n = 0;
    std::vector<int> byte_widths;
    std::size_t record_bytes = 0;
    std::size_t payload_bytes = 0;
    AnsZstdSpec ans;
};

struct ZUansSpec {
    std::size_t off = 0;
    std::size_t len = 0;
    int n = 0;
    std::size_t payload_bytes = 0;
    AnsZstdSpec ans;
};

struct TaggedSpec {
    std::size_t off = 0;
    std::size_t len = 0;
    int n = 0;
    int max_width = 1;
    BansSpec tag_bans;
    std::size_t value_payload_off = 0;
    std::size_t value_payload_bytes = 0;
};

struct ZTaggedSpec {
    std::size_t off = 0;
    std::size_t len = 0;
    int n = 0;
    int max_width = 1;
    ZBansSpec tags;
    std::size_t value_payload_bytes = 0;
    ZstdSpec value_zstd;
};

enum class EncKind {
    EMPTY,
    FIXD_MATRIX,
    TAGGED,
    BANS_MATRIX,
    FIXD_U8,
    UANS,
    ZFIXD_MATRIX,
    ZTAGGED,
    ZANS_MATRIX,
    ZFIXD_U8,
    ZUANS
};

struct ComponentSpec {
    std::string name;
    std::size_t payload_off = 0;
    std::size_t payload_len = 0;
    EncKind kind = EncKind::EMPTY;
    std::vector<FixedStreamSpec> fixed_streams;
    std::vector<BansSpec> bans_streams;
    TaggedSpec tagged;
    UansSpec uans;
    ZFixedMatrixSpec zfixed_matrix;
    ZFixedStreamSpec zfixed_stream;
    ZTaggedSpec ztagged;
    ZAnsMatrixSpec zans_matrix;
    ZUansSpec zuans;
};

struct HeaderSpec {
    std::uint32_t version = 0;
    int D = 0;
    int N = 0;
    std::int64_t p = 0;
    std::int64_t threshold = 0;
    std::int64_t diff = 0;
    double eb = 0.0;
    int num_block = 0;
    int num_idrel = 0;
    std::vector<double> mins;
    std::size_t mins_off = 0;
    ComponentSpec blk, idr, rel, cho;
};

static std::size_t fixed_payload_bytes(int n, int width) {
    if (n < 0 || width <= 0) throw std::runtime_error("invalid fixed size");
    return (static_cast<std::size_t>(n) * static_cast<std::size_t>(width) + 7) / 8;
}

static std::size_t checked_add_size(std::size_t a, std::size_t b, const char* name) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::runtime_error(std::string(name) + " size overflow");
    }
    return a + b;
}

static std::size_t checked_mul_size(std::size_t a, std::size_t b, const char* name) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(std::string(name) + " size overflow");
    }
    return a * b;
}

static std::size_t checked_align_up_size(std::size_t x, std::size_t a, const char* name) {
    if (a == 0 || x > std::numeric_limits<std::size_t>::max() - (a - 1)) {
        throw std::runtime_error(std::string(name) + " alignment overflow");
    }
    return align_up_size(x, a);
}

static ChunkedCodecSpec parse_chunked_codec_at(
    const std::vector<unsigned char>& buf,
    std::size_t off,
    const char magic[4]
) {
    if (off > buf.size()) throw std::runtime_error("chunked codec offset past end");
    Reader r(buf.data() + off, buf.size() - off);
    r.expect_magic(magic);
    ChunkedCodecSpec s;
    s.off = off;
    s.total_uncomp = u64_to_size(r.read_le<std::uint64_t>(), "chunked total_uncomp");
    s.max_chunk = u64_to_size(r.read_le<std::uint64_t>(), "chunked max_chunk");
    s.num_chunks = r.read_le<std::uint32_t>();
    std::size_t uncomp_total = 0;
    std::size_t comp_total = 0;
    std::size_t comp_total_aligned = 0;
    s.uncomp_chunk_bytes.reserve(s.num_chunks);
    s.comp_chunk_bytes.reserve(s.num_chunks);
    for (std::uint32_t c = 0; c < s.num_chunks; ++c) {
        std::size_t ub = u64_to_size(r.read_le<std::uint64_t>(), "chunked uncomp chunk bytes");
        std::size_t cb = u64_to_size(r.read_le<std::uint64_t>(), "chunked comp chunk bytes");
        if (ub > s.max_chunk) throw std::runtime_error("chunk exceeds recorded max_chunk");
        uncomp_total = checked_add_size(uncomp_total, ub, "chunked uncompressed total");
        comp_total = checked_add_size(comp_total, cb, "chunked compressed total");
        comp_total_aligned = checked_align_up_size(
            comp_total_aligned, NVCOMP_CHUNK_PTR_ALIGNMENT, "chunked compressed storage"
        );
        comp_total_aligned = checked_add_size(
            comp_total_aligned, cb, "aligned chunked compressed storage"
        );
        s.uncomp_chunk_bytes.push_back(ub);
        s.comp_chunk_bytes.push_back(cb);
    }
    if (uncomp_total != s.total_uncomp) {
        throw std::runtime_error("chunked uncompressed byte count mismatch");
    }
    s.comp_total = comp_total;
    s.comp_total_aligned = comp_total_aligned;
    s.data_off = off + r.pos;
    r.skip(comp_total);
    s.len = r.pos;
    if (s.total_uncomp == 0) {
        if (s.max_chunk != 0 || s.num_chunks != 0 || s.comp_total != 0) {
            throw std::runtime_error("bad empty chunked codec wrapper");
        }
    } else if (s.max_chunk == 0 || s.num_chunks == 0) {
        throw std::runtime_error("bad non-empty chunked codec wrapper");
    }
    return s;
}

static Nva1Spec parse_nva1_at(const std::vector<unsigned char>& buf, std::size_t off) {
    Nva1Spec s;
    static_cast<ChunkedCodecSpec&>(s) = parse_chunked_codec_at(buf, off, "NVA1");
    return s;
}

static ZstdSpec parse_zstg_at(const std::vector<unsigned char>& buf, std::size_t off) {
    ZstdSpec s;
    static_cast<ChunkedCodecSpec&>(s) = parse_chunked_codec_at(buf, off, "ZSTG");
    return s;
}

static AnsZstdSpec parse_ans_zstd_metadata_at(
    const std::vector<unsigned char>& buf,
    std::size_t off
) {
    if (off > buf.size()) throw std::runtime_error("ANS metadata offset past end");
    Reader r(buf.data() + off, buf.size() - off);
    AnsZstdSpec s;
    s.metadata_off = off;
    s.total_uncomp = u64_to_size(r.read_le<std::uint64_t>(), "ANS total_uncomp");
    s.max_chunk = u64_to_size(r.read_le<std::uint64_t>(), "ANS max_chunk");
    s.num_chunks = r.read_le<std::uint32_t>();
    std::size_t uncomp_total = 0;
    std::size_t comp_total = 0;
    std::size_t comp_total_aligned = 0;
    s.uncomp_chunk_bytes.reserve(s.num_chunks);
    s.comp_chunk_bytes.reserve(s.num_chunks);
    for (std::uint32_t c = 0; c < s.num_chunks; ++c) {
        std::size_t ub = u64_to_size(r.read_le<std::uint64_t>(), "ANS uncomp chunk bytes");
        std::size_t cb = u64_to_size(r.read_le<std::uint64_t>(), "ANS comp chunk bytes");
        if (ub > s.max_chunk) throw std::runtime_error("ANS chunk exceeds max_chunk");
        uncomp_total = checked_add_size(uncomp_total, ub, "ANS uncompressed total");
        comp_total = checked_add_size(comp_total, cb, "ANS compressed total");
        comp_total_aligned = checked_align_up_size(
            comp_total_aligned, NVCOMP_CHUNK_PTR_ALIGNMENT, "ANS compressed storage"
        );
        comp_total_aligned = checked_add_size(
            comp_total_aligned, cb, "aligned ANS compressed storage"
        );
        s.uncomp_chunk_bytes.push_back(ub);
        s.comp_chunk_bytes.push_back(cb);
    }
    if (uncomp_total != s.total_uncomp) {
        throw std::runtime_error("ANS uncompressed byte count mismatch");
    }
    std::size_t recorded_comp_total = u64_to_size(
        r.read_le<std::uint64_t>(), "ANS compressed total"
    );
    if (recorded_comp_total != comp_total) {
        throw std::runtime_error("ANS compressed byte count mismatch");
    }
    s.comp_total = comp_total;
    s.comp_total_aligned = comp_total_aligned;
    std::size_t zstg_len = u64_to_size(r.read_le<std::uint64_t>(), "ANS ZSTG length");
    s.zstd = parse_zstg_at(buf, off + r.pos);
    if (s.zstd.len != zstg_len) throw std::runtime_error("ANS ZSTG length mismatch");
    if (s.zstd.total_uncomp != s.comp_total) {
        throw std::runtime_error("ANS ZSTG output size mismatch");
    }
    r.skip(zstg_len);
    s.len = r.pos;
    if (s.total_uncomp == 0) {
        if (s.max_chunk != 0 || s.num_chunks != 0 || s.comp_total != 0) {
            throw std::runtime_error("bad empty ANS metadata");
        }
    } else if (s.max_chunk == 0 || s.num_chunks == 0) {
        throw std::runtime_error("bad non-empty ANS metadata");
    }
    return s;
}

static BansSpec parse_bans_at(const std::vector<unsigned char>& buf, std::size_t off) {
    Reader r(buf.data() + off, buf.size() - off);
    r.expect_magic("BANS");
    BansSpec s; s.off = off;
    s.byte_width = static_cast<int>(r.read_u8());
    if (s.byte_width < 1 || s.byte_width > 4) throw std::runtime_error("bad BANS byte_width");
    s.n = u64_to_int(r.read_le<std::uint64_t>(), "BANS n");
    std::size_t ans_len = u64_to_size(r.read_le<std::uint64_t>(), "BANS ans len");
    std::size_t nva_off = off + r.pos;
    s.nva = parse_nva1_at(buf, nva_off);
    if (s.nva.len != ans_len) throw std::runtime_error("BANS ans_blob_length mismatch");
    if (s.nva.total_uncomp != static_cast<std::size_t>(s.n) * static_cast<std::size_t>(s.byte_width)) throw std::runtime_error("BANS uncompressed byte count mismatch");
    s.len = r.pos + ans_len;
    return s;
}

static UansSpec parse_uans_at(const std::vector<unsigned char>& buf, std::size_t off) {
    Reader r(buf.data() + off, buf.size() - off);
    r.expect_magic("UANS");
    UansSpec s; s.off = off;
    s.n = u64_to_int(r.read_le<std::uint64_t>(), "UANS n");
    std::size_t ans_len = u64_to_size(r.read_le<std::uint64_t>(), "UANS ans len");
    std::size_t nva_off = off + r.pos;
    s.nva = parse_nva1_at(buf, nva_off);
    if (s.nva.len != ans_len) throw std::runtime_error("UANS ans_blob_length mismatch");
    if (s.nva.total_uncomp != static_cast<std::size_t>(s.n)) throw std::runtime_error("UANS uncompressed byte count mismatch");
    s.len = r.pos + ans_len;
    return s;
}

static FixedStreamSpec parse_fixed_at(const std::vector<unsigned char>& buf, std::size_t off) {
    Reader r(buf.data() + off, buf.size() - off);
    r.expect_magic("FIXD");
    FixedStreamSpec s; s.off = off;
    s.mn = r.read_le<std::int64_t>();
    s.n = u64_to_int(r.read_le<std::uint64_t>(), "FIXD n");
    s.width = static_cast<int>(r.read_u8());
    if (s.width <= 0 || s.width > 32) throw std::runtime_error("bad FIXD width for int32 codec");
    s.payload_off = off + r.pos;
    s.payload_bytes = fixed_payload_bytes(s.n, s.width);
    r.skip(s.payload_bytes);
    s.len = r.pos;
    return s;
}

static ZFixedStreamSpec parse_zfixed_at(
    const std::vector<unsigned char>& buf,
    std::size_t off
) {
    if (off > buf.size()) throw std::runtime_error("ZFXD offset past end");
    Reader r(buf.data() + off, buf.size() - off);
    r.expect_magic("ZFXD");
    ZFixedStreamSpec s;
    s.off = off;
    s.field.mn = r.read_le<std::int64_t>();
    s.field.n = u64_to_int(r.read_le<std::uint64_t>(), "ZFXD n");
    s.field.width = static_cast<int>(r.read_u8());
    if (s.field.width <= 0 || s.field.width > 32) {
        throw std::runtime_error("bad ZFXD width");
    }
    s.field.payload_bytes = u64_to_size(r.read_le<std::uint64_t>(), "ZFXD raw payload bytes");
    if (s.field.payload_bytes != fixed_payload_bytes(s.field.n, s.field.width)) {
        throw std::runtime_error("ZFXD raw payload byte count mismatch");
    }
    std::size_t zstg_len = u64_to_size(r.read_le<std::uint64_t>(), "ZFXD ZSTG length");
    s.zstd = parse_zstg_at(buf, off + r.pos);
    if (s.zstd.len != zstg_len) throw std::runtime_error("ZFXD ZSTG length mismatch");
    if (s.zstd.total_uncomp != s.field.payload_bytes) {
        throw std::runtime_error("ZFXD ZSTG output size mismatch");
    }
    r.skip(zstg_len);
    s.len = r.pos;
    return s;
}

static ZFixedMatrixSpec parse_zfmx_at(
    const std::vector<unsigned char>& buf,
    std::size_t off
) {
    if (off > buf.size()) throw std::runtime_error("ZFMX offset past end");
    Reader r(buf.data() + off, buf.size() - off);
    r.expect_magic("ZFMX");
    ZFixedMatrixSpec s;
    s.off = off;
    std::uint32_t Du = r.read_le<std::uint32_t>();
    if (Du > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("ZFMX D too large");
    }
    s.D = static_cast<int>(Du);
    s.n = u64_to_int(r.read_le<std::uint64_t>(), "ZFMX n");
    if (s.D <= 0) throw std::runtime_error("bad ZFMX D");
    s.fields.reserve(static_cast<std::size_t>(s.D));
    std::size_t raw_total = 0;
    for (int d = 0; d < s.D; ++d) {
        FixedFieldSpec f;
        f.mn = r.read_le<std::int64_t>();
        f.n = s.n;
        f.width = static_cast<int>(r.read_u8());
        if (f.width <= 0 || f.width > 32) throw std::runtime_error("bad ZFMX width");
        f.payload_offset = raw_total;
        f.payload_bytes = u64_to_size(r.read_le<std::uint64_t>(), "ZFMX raw stream bytes");
        if (f.payload_bytes != fixed_payload_bytes(f.n, f.width)) {
            throw std::runtime_error("ZFMX per-dimension payload byte count mismatch");
        }
        raw_total = checked_add_size(raw_total, f.payload_bytes, "ZFMX raw payload");
        s.fields.push_back(f);
    }
    s.raw_payload_bytes = u64_to_size(r.read_le<std::uint64_t>(), "ZFMX total raw bytes");
    if (s.raw_payload_bytes != raw_total) throw std::runtime_error("ZFMX raw total mismatch");
    std::size_t zstg_len = u64_to_size(r.read_le<std::uint64_t>(), "ZFMX ZSTG length");
    s.zstd = parse_zstg_at(buf, off + r.pos);
    if (s.zstd.len != zstg_len) throw std::runtime_error("ZFMX ZSTG length mismatch");
    if (s.zstd.total_uncomp != s.raw_payload_bytes) {
        throw std::runtime_error("ZFMX ZSTG output size mismatch");
    }
    r.skip(zstg_len);
    s.len = r.pos;
    return s;
}

static ZBansSpec parse_zbns_at(const std::vector<unsigned char>& buf, std::size_t off) {
    if (off > buf.size()) throw std::runtime_error("ZBNS offset past end");
    Reader r(buf.data() + off, buf.size() - off);
    r.expect_magic("ZBNS");
    ZBansSpec s;
    s.off = off;
    s.byte_width = static_cast<int>(r.read_u8());
    if (s.byte_width < 1 || s.byte_width > 4) throw std::runtime_error("bad ZBNS byte_width");
    s.n = u64_to_int(r.read_le<std::uint64_t>(), "ZBNS n");
    s.payload_bytes = u64_to_size(r.read_le<std::uint64_t>(), "ZBNS payload bytes");
    std::size_t expected = checked_mul_size(
        static_cast<std::size_t>(s.n), static_cast<std::size_t>(s.byte_width), "ZBNS payload"
    );
    if (s.payload_bytes != expected) throw std::runtime_error("ZBNS payload byte count mismatch");
    s.ans = parse_ans_zstd_metadata_at(buf, off + r.pos);
    if (s.ans.total_uncomp != s.payload_bytes) throw std::runtime_error("ZBNS ANS output size mismatch");
    r.skip(s.ans.len);
    s.len = r.pos;
    return s;
}

static ZAnsMatrixSpec parse_zamx_at(const std::vector<unsigned char>& buf, std::size_t off) {
    if (off > buf.size()) throw std::runtime_error("ZAMX offset past end");
    Reader r(buf.data() + off, buf.size() - off);
    r.expect_magic("ZAMX");
    ZAnsMatrixSpec s;
    s.off = off;
    std::uint32_t Du = r.read_le<std::uint32_t>();
    if (Du > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("ZAMX D too large");
    }
    s.D = static_cast<int>(Du);
    s.n = u64_to_int(r.read_le<std::uint64_t>(), "ZAMX n");
    if (s.D <= 0) throw std::runtime_error("bad ZAMX D");
    s.byte_widths.reserve(static_cast<std::size_t>(s.D));
    for (int d = 0; d < s.D; ++d) {
        int width = static_cast<int>(r.read_u8());
        if (width < 1 || width > 4) throw std::runtime_error("bad ZAMX byte_width");
        s.record_bytes = checked_add_size(
            s.record_bytes, static_cast<std::size_t>(width), "ZAMX record"
        );
        s.byte_widths.push_back(width);
    }
    s.payload_bytes = u64_to_size(r.read_le<std::uint64_t>(), "ZAMX payload bytes");
    std::size_t expected = checked_mul_size(
        static_cast<std::size_t>(s.n), s.record_bytes, "ZAMX payload"
    );
    if (s.payload_bytes != expected) throw std::runtime_error("ZAMX payload byte count mismatch");
    s.ans = parse_ans_zstd_metadata_at(buf, off + r.pos);
    if (s.ans.total_uncomp != s.payload_bytes) throw std::runtime_error("ZAMX ANS output size mismatch");
    r.skip(s.ans.len);
    s.len = r.pos;
    return s;
}

static ZUansSpec parse_zuns_at(const std::vector<unsigned char>& buf, std::size_t off) {
    if (off > buf.size()) throw std::runtime_error("ZUNS offset past end");
    Reader r(buf.data() + off, buf.size() - off);
    r.expect_magic("ZUNS");
    ZUansSpec s;
    s.off = off;
    s.n = u64_to_int(r.read_le<std::uint64_t>(), "ZUNS n");
    s.payload_bytes = u64_to_size(r.read_le<std::uint64_t>(), "ZUNS payload bytes");
    if (s.payload_bytes != static_cast<std::size_t>(s.n)) {
        throw std::runtime_error("ZUNS payload byte count mismatch");
    }
    s.ans = parse_ans_zstd_metadata_at(buf, off + r.pos);
    if (s.ans.total_uncomp != s.payload_bytes) throw std::runtime_error("ZUNS ANS output size mismatch");
    r.skip(s.ans.len);
    s.len = r.pos;
    return s;
}

static TaggedSpec parse_tag3_at(const std::vector<unsigned char>& buf, std::size_t off, std::size_t payload_len) {
    Reader r(buf.data() + off, payload_len);
    r.expect_magic("TAG3");
    TaggedSpec s; s.off = off; s.len = payload_len;
    s.n = u64_to_int(r.read_le<std::uint64_t>(), "TAG3 n");
    s.max_width = static_cast<int>(r.read_u8());
    std::size_t tag_len = u64_to_size(r.read_le<std::uint64_t>(), "TAG3 tag_payload_len");
    if (tag_len > 0) {
        s.tag_bans = parse_bans_at(buf, off + r.pos);
        if (s.tag_bans.len != tag_len) throw std::runtime_error("TAG3 tag payload length mismatch");
        if (s.tag_bans.n != s.n) throw std::runtime_error("TAG3 tag count mismatch");
    }
    r.skip(tag_len);
    s.value_payload_off = off + r.pos;
    s.value_payload_bytes = r.remaining();
    return s;
}

static ZTaggedSpec parse_ztg3_at(
    const std::vector<unsigned char>& buf,
    std::size_t off,
    std::size_t payload_len
) {
    Reader r(buf.data() + off, payload_len);
    r.expect_magic("ZTG3");
    ZTaggedSpec s;
    s.off = off;
    s.len = payload_len;
    s.n = u64_to_int(r.read_le<std::uint64_t>(), "ZTG3 n");
    s.max_width = static_cast<int>(r.read_u8());
    if (s.max_width < 1 || s.max_width > 32) throw std::runtime_error("bad ZTG3 max_width");
    std::size_t tag_len = u64_to_size(r.read_le<std::uint64_t>(), "ZTG3 tag payload length");
    s.tags = parse_zbns_at(buf, off + r.pos);
    if (s.tags.len != tag_len) throw std::runtime_error("ZTG3 tag payload length mismatch");
    if (s.tags.n != s.n) throw std::runtime_error("ZTG3 tag count mismatch");
    r.skip(tag_len);
    s.value_payload_bytes = u64_to_size(r.read_le<std::uint64_t>(), "ZTG3 value bytes");
    std::size_t zstg_len = u64_to_size(r.read_le<std::uint64_t>(), "ZTG3 value ZSTG length");
    s.value_zstd = parse_zstg_at(buf, off + r.pos);
    if (s.value_zstd.len != zstg_len) throw std::runtime_error("ZTG3 value ZSTG length mismatch");
    if (s.value_zstd.total_uncomp != s.value_payload_bytes) {
        throw std::runtime_error("ZTG3 value ZSTG output size mismatch");
    }
    r.skip(zstg_len);
    if (r.remaining() != 0) throw std::runtime_error("trailing bytes in ZTG3 payload");
    return s;
}

static ComponentSpec parse_component_at(const std::vector<unsigned char>& buf, Reader& top, const char expected[4], int D, int n, bool is_choice=false) {
    ComponentSpec c;
    c.name = top.read_magic();
    if (c.name != std::string(expected, 4)) throw std::runtime_error("component order/name mismatch");
    c.payload_len = u64_to_size(top.read_le<std::uint64_t>(), "component payload len");
    c.payload_off = top.pos;
    top.skip(c.payload_len);
    if (c.payload_len == 5 && std::memcmp(buf.data() + c.payload_off, "EMPTY", 5) == 0) { c.kind = EncKind::EMPTY; return c; }
    if (c.payload_len < 4) throw std::runtime_error("component payload too small");
    std::string magic(reinterpret_cast<const char*>(buf.data() + c.payload_off), 4);
    if (is_choice) {
        if (magic == "FIXD") {
            c.kind = EncKind::FIXD_U8;
            c.fixed_streams.push_back(parse_fixed_at(buf, c.payload_off));
            if (c.fixed_streams[0].len != c.payload_len) throw std::runtime_error("CHO0 FIXD length mismatch");
        } else if (magic == "UANS") {
            c.kind = EncKind::UANS;
            c.uans = parse_uans_at(buf, c.payload_off);
            if (c.uans.len != c.payload_len) throw std::runtime_error("CHO0 UANS length mismatch");
        } else if (magic == "ZFXD") {
            c.kind = EncKind::ZFIXD_U8;
            c.zfixed_stream = parse_zfixed_at(buf, c.payload_off);
            if (c.zfixed_stream.len != c.payload_len) throw std::runtime_error("CHO0 ZFXD length mismatch");
            if (c.zfixed_stream.field.n != n || c.zfixed_stream.field.width > 8) {
                throw std::runtime_error("CHO0 ZFXD metadata mismatch");
            }
        } else if (magic == "ZUNS") {
            c.kind = EncKind::ZUANS;
            c.zuans = parse_zuns_at(buf, c.payload_off);
            if (c.zuans.len != c.payload_len || c.zuans.n != n) {
                throw std::runtime_error("CHO0 ZUNS metadata mismatch");
            }
        }
        else throw std::runtime_error("unsupported CHO0 encoding " + magic);
        return c;
    }
    if (magic == "FIXD") {
        c.kind = EncKind::FIXD_MATRIX;
        std::size_t p = c.payload_off;
        for (int d = 0; d < D; ++d) {
            FixedStreamSpec fs = parse_fixed_at(buf, p);
            if (fs.n != n) throw std::runtime_error("FIXD matrix stream length mismatch");
            c.fixed_streams.push_back(fs);
            p += fs.len;
        }
        if (p != c.payload_off + c.payload_len) throw std::runtime_error("trailing bytes in FIXD matrix component");
    } else if (magic == "TAG3") {
        c.kind = EncKind::TAGGED;
        c.tagged = parse_tag3_at(buf, c.payload_off, c.payload_len);
        if (c.tagged.n != n) throw std::runtime_error("TAG3 component n mismatch");
    } else if (magic == "BANS") {
        c.kind = EncKind::BANS_MATRIX;
        std::size_t p = c.payload_off;
        for (int d = 0; d < D; ++d) {
            BansSpec bs = parse_bans_at(buf, p);
            if (bs.n != n) throw std::runtime_error("BANS matrix stream length mismatch");
            c.bans_streams.push_back(bs);
            p += bs.len;
        }
        if (p != c.payload_off + c.payload_len) throw std::runtime_error("trailing bytes in BANS matrix component");
    } else if (magic == "ZFMX") {
        c.kind = EncKind::ZFIXD_MATRIX;
        c.zfixed_matrix = parse_zfmx_at(buf, c.payload_off);
        if (c.zfixed_matrix.len != c.payload_len || c.zfixed_matrix.D != D || c.zfixed_matrix.n != n) {
            throw std::runtime_error("ZFMX component metadata mismatch");
        }
    } else if (magic == "ZTG3") {
        c.kind = EncKind::ZTAGGED;
        c.ztagged = parse_ztg3_at(buf, c.payload_off, c.payload_len);
        if (c.ztagged.n != n) throw std::runtime_error("ZTG3 component n mismatch");
    } else if (magic == "ZAMX") {
        c.kind = EncKind::ZANS_MATRIX;
        c.zans_matrix = parse_zamx_at(buf, c.payload_off);
        if (c.zans_matrix.len != c.payload_len || c.zans_matrix.D != D || c.zans_matrix.n != n) {
            throw std::runtime_error("ZAMX component metadata mismatch");
        }
    } else if (magic == "ZBNS" && D == 1) {
        c.kind = EncKind::ZANS_MATRIX;
        ZBansSpec z = parse_zbns_at(buf, c.payload_off);
        if (z.len != c.payload_len || z.n != n) throw std::runtime_error("ZBNS component metadata mismatch");
        c.zans_matrix.off = z.off;
        c.zans_matrix.len = z.len;
        c.zans_matrix.D = 1;
        c.zans_matrix.n = z.n;
        c.zans_matrix.byte_widths.push_back(z.byte_width);
        c.zans_matrix.record_bytes = static_cast<std::size_t>(z.byte_width);
        c.zans_matrix.payload_bytes = z.payload_bytes;
        c.zans_matrix.ans = z.ans;
    } else if (magic == "ZFXD" && D == 1) {
        c.kind = EncKind::ZFIXD_MATRIX;
        c.zfixed_stream = parse_zfixed_at(buf, c.payload_off);
        if (c.zfixed_stream.len != c.payload_len || c.zfixed_stream.field.n != n) {
            throw std::runtime_error("ZFXD component metadata mismatch");
        }
        c.zfixed_matrix.off = c.zfixed_stream.off;
        c.zfixed_matrix.len = c.zfixed_stream.len;
        c.zfixed_matrix.D = 1;
        c.zfixed_matrix.n = n;
        c.zfixed_matrix.fields.push_back(c.zfixed_stream.field);
        c.zfixed_matrix.raw_payload_bytes = c.zfixed_stream.field.payload_bytes;
        c.zfixed_matrix.zstd = c.zfixed_stream.zstd;
    } else {
        throw std::runtime_error("unsupported component encoding " + magic);
    }
    return c;
}

static HeaderSpec parse_lcg1(const std::vector<unsigned char>& buf) {
    Reader r(buf.data(), buf.size());
    r.expect_magic("LCG1");
    HeaderSpec h;
    h.version = r.read_le<std::uint32_t>();
    if (h.version != 1) throw std::runtime_error("unsupported LCG1 version");
    std::uint32_t Du = r.read_le<std::uint32_t>();
    if (Du > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("D too large");
    }
    h.D = static_cast<int>(Du);
    h.N = u64_to_int(r.read_le<std::uint64_t>(), "N");
    h.p = r.read_le<std::int64_t>();
    h.threshold = r.read_le<std::int64_t>();
    h.diff = r.read_le<std::int64_t>();
    h.eb = r.read_le<double>();
    h.num_block = u64_to_int(r.read_le<std::uint64_t>(), "num_block");
    h.num_idrel = u64_to_int(r.read_le<std::uint64_t>(), "num_idrel");
    if (h.D <= 0 || h.N <= 0 || h.p <= 0
        || h.p > static_cast<std::int64_t>(std::numeric_limits<int>::max())
        || !std::isfinite(h.eb) || h.eb <= 0.0) {
        throw std::runtime_error("bad LCG1 metadata");
    }
    if (h.num_block < 0 || h.num_idrel < 0
        || h.num_block > h.N || h.num_idrel > h.N
        || static_cast<std::int64_t>(h.num_block) + h.num_idrel != h.N) {
        throw std::runtime_error("LCG1 compact-stream counts do not sum to N");
    }
    if (static_cast<std::size_t>(h.D) > static_cast<std::size_t>(std::numeric_limits<int>::max()) / static_cast<std::size_t>(h.N)) {
        throw std::runtime_error("D*N exceeds int32 kernel indexing range");
    }
    h.mins.resize(h.D);
    h.mins_off = r.pos;
    for (int d = 0; d < h.D; ++d) {
        h.mins[d] = r.read_le<double>();
        if (!std::isfinite(h.mins[d])) throw std::runtime_error("non-finite LCG1 minimum");
    }
    h.blk = parse_component_at(buf, r, "BLK0", h.D, h.num_block, false);
    h.idr = parse_component_at(buf, r, "IDR0", 1, h.num_idrel, false);
    h.rel = parse_component_at(buf, r, "REL0", h.D, h.N, false);
    h.cho = parse_component_at(buf, r, "CHO0", 1, h.N, true);
    if (r.remaining() != 0) throw std::runtime_error("trailing bytes after LCG1 container");
    return h;
}

// -------------------------- global codec planning --------------------------
// Every Zstd chunk from every component is submitted in one nvCOMP batch.
// Every ANS chunk (legacy or Zstd-wrapped) is likewise submitted in one later
// batch.  The two batches share a stream, so Zstd -> ANS dependencies require
// no host synchronization.
struct AnsBatchSpecRef {
    Nva1Spec* legacy = nullptr;
    AnsZstdSpec* wrapped = nullptr;

    explicit AnsBatchSpecRef(Nva1Spec* s) : legacy(s) {}
    explicit AnsBatchSpecRef(AnsZstdSpec* s) : wrapped(s) {}
};

struct CodecBatchPlan {
    std::vector<ZstdSpec*> zstd_specs;
    std::vector<AnsBatchSpecRef> ans_specs;

    std::size_t zstd_num_chunks = 0;
    std::size_t zstd_max_chunk = 0;
    std::size_t zstd_total_uncomp = 0;
    std::size_t zstd_output_bytes = 0;
    std::size_t zstd_aligned_input_bytes = 0;

    std::size_t ans_num_chunks = 0;
    std::size_t ans_max_chunk = 0;
    std::size_t ans_total_uncomp = 0;
    std::size_t ans_output_bytes = 0;
    std::size_t ans_aligned_input_bytes = 0;

    int max_tag_n = 0;
};

static void validate_chunk_vectors(
    std::uint32_t num_chunks,
    const std::vector<std::size_t>& uncomp,
    const std::vector<std::size_t>& comp,
    const char* codec
) {
    if (uncomp.size() != num_chunks || comp.size() != num_chunks) {
        throw std::runtime_error(std::string(codec) + " chunk metadata count mismatch");
    }
}

static void plan_zstd_spec(ZstdSpec& s, CodecBatchPlan& p) {
    if (s.total_uncomp == 0) {
        s.decoded_offset = 0;
        return;
    }
    if (s.decoded_offset != UNPLANNED_DEVICE_OFFSET) return;
    validate_chunk_vectors(s.num_chunks, s.uncomp_chunk_bytes, s.comp_chunk_bytes, "Zstd");

    p.zstd_output_bytes = checked_align_up_size(
        p.zstd_output_bytes, NVCOMP_CHUNK_PTR_ALIGNMENT, "Zstd output arena"
    );
    s.decoded_offset = p.zstd_output_bytes;
    p.zstd_output_bytes = checked_add_size(
        p.zstd_output_bytes, s.total_uncomp, "Zstd output arena"
    );
    p.zstd_num_chunks = checked_add_size(
        p.zstd_num_chunks, s.num_chunks, "Zstd batch chunk count"
    );
    p.zstd_total_uncomp = checked_add_size(
        p.zstd_total_uncomp, s.total_uncomp, "Zstd batch output"
    );
    p.zstd_max_chunk = std::max(p.zstd_max_chunk, s.max_chunk);
    for (std::size_t cb : s.comp_chunk_bytes) {
        p.zstd_aligned_input_bytes = checked_align_up_size(
            p.zstd_aligned_input_bytes,
            NVCOMP_CHUNK_PTR_ALIGNMENT,
            "Zstd aligned input arena"
        );
        p.zstd_aligned_input_bytes = checked_add_size(
            p.zstd_aligned_input_bytes, cb, "Zstd aligned input arena"
        );
    }
    p.zstd_specs.push_back(&s);
}

static std::size_t reserve_ans_output(std::size_t bytes, CodecBatchPlan& p) {
    p.ans_output_bytes = checked_align_up_size(
        p.ans_output_bytes, NVCOMP_CHUNK_PTR_ALIGNMENT, "ANS output arena"
    );
    std::size_t off = p.ans_output_bytes;
    p.ans_output_bytes = checked_add_size(
        p.ans_output_bytes, bytes, "ANS output arena"
    );
    return off;
}

static void account_ans_chunks(
    std::uint32_t num_chunks,
    std::size_t max_chunk,
    std::size_t total_uncomp,
    const std::vector<std::size_t>& comp_chunk_bytes,
    CodecBatchPlan& p
) {
    p.ans_num_chunks = checked_add_size(
        p.ans_num_chunks, num_chunks, "ANS batch chunk count"
    );
    p.ans_total_uncomp = checked_add_size(
        p.ans_total_uncomp, total_uncomp, "ANS batch output"
    );
    p.ans_max_chunk = std::max(p.ans_max_chunk, max_chunk);
    for (std::size_t cb : comp_chunk_bytes) {
        p.ans_aligned_input_bytes = checked_align_up_size(
            p.ans_aligned_input_bytes,
            NVCOMP_CHUNK_PTR_ALIGNMENT,
            "ANS aligned input arena"
        );
        p.ans_aligned_input_bytes = checked_add_size(
            p.ans_aligned_input_bytes, cb, "ANS aligned input arena"
        );
    }
}

static void plan_legacy_ans_spec(Nva1Spec& s, CodecBatchPlan& p) {
    if (s.total_uncomp == 0) {
        s.decoded_offset = 0;
        return;
    }
    if (s.decoded_offset != UNPLANNED_DEVICE_OFFSET) return;
    validate_chunk_vectors(s.num_chunks, s.uncomp_chunk_bytes, s.comp_chunk_bytes, "ANS");
    s.decoded_offset = reserve_ans_output(s.total_uncomp, p);
    account_ans_chunks(
        s.num_chunks, s.max_chunk, s.total_uncomp, s.comp_chunk_bytes, p
    );
    p.ans_specs.emplace_back(&s);
}

static void plan_wrapped_ans_spec(AnsZstdSpec& s, CodecBatchPlan& p) {
    if (s.total_uncomp == 0) {
        s.decoded_offset = 0;
        return;
    }
    if (s.decoded_offset != UNPLANNED_DEVICE_OFFSET) return;
    validate_chunk_vectors(s.num_chunks, s.uncomp_chunk_bytes, s.comp_chunk_bytes, "ANS");
    plan_zstd_spec(s.zstd, p);
    s.decoded_offset = reserve_ans_output(s.total_uncomp, p);
    account_ans_chunks(
        s.num_chunks, s.max_chunk, s.total_uncomp, s.comp_chunk_bytes, p
    );
    p.ans_specs.emplace_back(&s);
}

static void plan_component_codecs(ComponentSpec& c, CodecBatchPlan& p) {
    switch (c.kind) {
        case EncKind::BANS_MATRIX:
            for (BansSpec& bs : c.bans_streams) plan_legacy_ans_spec(bs.nva, p);
            break;
        case EncKind::TAGGED:
            plan_legacy_ans_spec(c.tagged.tag_bans.nva, p);
            p.max_tag_n = std::max(p.max_tag_n, c.tagged.n);
            break;
        case EncKind::UANS:
            plan_legacy_ans_spec(c.uans.nva, p);
            break;
        case EncKind::ZFIXD_MATRIX:
            plan_zstd_spec(c.zfixed_matrix.zstd, p);
            break;
        case EncKind::ZANS_MATRIX:
            plan_wrapped_ans_spec(c.zans_matrix.ans, p);
            break;
        case EncKind::ZTAGGED:
            plan_wrapped_ans_spec(c.ztagged.tags.ans, p);
            plan_zstd_spec(c.ztagged.value_zstd, p);
            p.max_tag_n = std::max(p.max_tag_n, c.ztagged.n);
            break;
        case EncKind::ZFIXD_U8:
            plan_zstd_spec(c.zfixed_stream.zstd, p);
            break;
        case EncKind::ZUANS:
            plan_wrapped_ans_spec(c.zuans.ans, p);
            break;
        case EncKind::EMPTY:
        case EncKind::FIXD_MATRIX:
        case EncKind::FIXD_U8:
            break;
    }
}

static CodecBatchPlan build_codec_batch_plan(HeaderSpec& h) {
    CodecBatchPlan p;
    plan_component_codecs(h.blk, p);
    plan_component_codecs(h.idr, p);
    plan_component_codecs(h.rel, p);
    plan_component_codecs(h.cho, p);
    return p;
}

class ArenaLayout {
public:
    std::size_t bytes = 0;

    std::size_t reserve_bytes(
        std::size_t n,
        std::size_t alignment,
        const char* name
    ) {
        bytes = checked_align_up_size(bytes, alignment, name);
        std::size_t off = bytes;
        bytes = checked_add_size(bytes, n, name);
        return off;
    }

    template <typename T>
    std::size_t reserve(std::size_t count, const char* name) {
        return reserve_bytes(
            checked_mul_size(count, sizeof(T), name), alignof(T), name
        );
    }
};

template <typename T>
static T* arena_ptr(unsigned char* base, std::size_t off) {
    return reinterpret_cast<T*>(base + off);
}

struct DeviceChunkCopyDesc {
    const unsigned char* src;
    unsigned char* dst;
    std::size_t bytes;
};
static_assert(
    std::is_trivially_copyable<DeviceChunkCopyDesc>::value,
    "chunk descriptors must be safe to upload with cudaMemcpyAsync"
);

// -------------------------- device-side helpers --------------------------
__device__ __forceinline__ std::uint64_t read_le_u64_dev(const unsigned char* p) {
    std::uint64_t v = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) v |= (static_cast<std::uint64_t>(p[i]) << (8 * i));
    return v;
}

__device__ __forceinline__ double read_le_double_dev(const unsigned char* p) {
    std::uint64_t u = read_le_u64_dev(p);
    double v;
    memcpy(&v, &u, sizeof(double));
    return v;
}

// nvCOMP may issue aligned vectorized loads from each compressed chunk pointer.
// The host builds these descriptors after the one device arena is allocated.
// One block copies one byte-tight chunk into aligned device storage.  The same
// storage is reused for the later ANS batch because both stages share a stream.
__global__ void copy_codec_chunks_to_aligned_kernel(
    const DeviceChunkCopyDesc* __restrict__ descs,
    std::size_t num_chunks
) {
    std::size_t c = static_cast<std::size_t>(blockIdx.x);
    if (c >= num_chunks) return;
    const DeviceChunkCopyDesc& d = descs[c];
    for (std::size_t j = threadIdx.x; j < d.bytes; j += blockDim.x) {
        d.dst[j] = d.src[j];
    }
}

__global__ void unpack_minbytes_i32_kernel(
    const unsigned char* __restrict__ packed,
    int n,
    int byte_width,
    int* __restrict__ out
) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    std::uint32_t v = 0;
    std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(byte_width);
    for (int b = 0; b < byte_width; ++b) v |= static_cast<std::uint32_t>(packed[base + b]) << (8 * b);
    out[i] = static_cast<int>(v);
}

__global__ void unpack_point_major_i32_dimension_kernel(
    const unsigned char* __restrict__ packed,
    int n,
    int byte_width,
    std::size_t record_bytes,
    std::size_t dimension_byte_offset,
    int* __restrict__ out
) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    std::size_t base = static_cast<std::size_t>(i) * record_bytes
        + dimension_byte_offset;
    std::uint32_t v = 0;
    for (int b = 0; b < byte_width; ++b) {
        v |= static_cast<std::uint32_t>(packed[base + static_cast<std::size_t>(b)])
            << (8 * b);
    }
    out[i] = static_cast<int>(v);
}

__global__ void unpack_fixed_i32_kernel(
    const unsigned char* __restrict__ payload,
    int n,
    int width,
    int mn,
    int* __restrict__ out
) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    std::size_t bitpos = static_cast<std::size_t>(i) * static_cast<std::size_t>(width);
    std::uint32_t v = 0;
    for (int b = 0; b < width; ++b) {
        std::size_t g = bitpos + static_cast<std::size_t>(b);
        std::size_t byte_idx = g >> 3;
        int bit_in_byte = 7 - static_cast<int>(g & 7ULL);
        v = (v << 1) | ((payload[byte_idx] >> bit_in_byte) & 1u);
    }
    out[i] = mn + static_cast<int>(v);
}

__global__ void unpack_fixed_u8_kernel(
    const unsigned char* __restrict__ payload,
    int n,
    int width,
    int mn,
    unsigned char* __restrict__ out
) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    std::size_t bitpos = static_cast<std::size_t>(i) * static_cast<std::size_t>(width);
    std::uint32_t v = 0;
    for (int b = 0; b < width; ++b) {
        std::size_t g = bitpos + static_cast<std::size_t>(b);
        std::size_t byte_idx = g >> 3;
        int bit_in_byte = 7 - static_cast<int>(g & 7ULL);
        v = (v << 1) | ((payload[byte_idx] >> bit_in_byte) & 1u);
    }
    out[i] = static_cast<unsigned char>(mn + static_cast<int>(v));
}

__global__ void compute_tag_bits_from_tags_kernel(const int* tags, int D, int n, unsigned long long* bits) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bits[i] = static_cast<unsigned long long>(tags[i]) * static_cast<unsigned long long>(D);
}

__global__ void unpack_tagged_values_i32_kernel(
    const unsigned char* __restrict__ payload,
    const int* __restrict__ tags,
    const unsigned long long* __restrict__ offsets,
    int D,
    int n,
    int stride,
    int* __restrict__ out
) {
    int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    int total = D * n;
    if (idx >= total) return;
    int d = idx / n;
    int i = idx - d * n;
    int tag = tags[i];
    unsigned long long bitpos = offsets[i] + static_cast<unsigned long long>(d * tag);
    std::uint32_t v = 0;
    for (int b = 0; b < tag; ++b) {
        unsigned long long g = bitpos + static_cast<unsigned long long>(b);
        std::size_t byte_idx = static_cast<std::size_t>(g >> 3);
        int bit_in_byte = 7 - static_cast<int>(g & 7ULL);
        v = (v << 1) | ((payload[byte_idx] >> bit_in_byte) & 1u);
    }
    out[static_cast<std::size_t>(d) * stride + i] = static_cast<int>(v);
}

__global__ void flags_from_choice_kernel(
    const unsigned char* __restrict__ choice,
    int N,
    int* __restrict__ flag_idrel,
    int* __restrict__ flag_block
) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int is_idrel = choice[i] ? 1 : 0;
    flag_idrel[i] = is_idrel;
    flag_block[i] = 1 - is_idrel;
}

__global__ void init_bid_and_parent_kernel(
    const unsigned char* __restrict__ choice,
    const int* __restrict__ pos_idrel,
    const int* __restrict__ pos_block,
    const int* __restrict__ outIDRel,
    const int* __restrict__ outBlock,
    int N,
    int D,
    int num_block,
    int* __restrict__ bid,
    int* __restrict__ parent
) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (choice[i]) {
        int j = pos_idrel[i];
        parent[i] = i + outIDRel[j];
    } else {
        int j = pos_block[i];
        parent[i] = i;
        for (int d = 0; d < D; ++d) {
            bid[static_cast<std::size_t>(d) * N + i] = outBlock[static_cast<std::size_t>(d) * num_block + j];
        }
    }
}

__global__ void pointer_jump_kernel(const int* __restrict__ parent_in, int* __restrict__ parent_out, int N) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int p = parent_in[i];
    parent_out[i] = parent_in[p];
}

__global__ void fill_bid_from_root_kernel(
    int* __restrict__ bid,
    const int* __restrict__ root,
    int N,
    int D
) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(N) * D;
    if (idx >= total) return;
    int d = static_cast<int>(idx / static_cast<std::size_t>(N));
    int i = static_cast<int>(idx - static_cast<std::size_t>(d) * N);
    int r = root[i];
    bid[idx] = bid[static_cast<std::size_t>(d) * N + r];
}

__global__ void reconstruct_coords_kernel(
    const int* __restrict__ bid,
    const int* __restrict__ rel,
    const unsigned char* __restrict__ comp,
    std::size_t mins_off,
    int N,
    int D,
    long long p,
    double eb,
    bool output_soa,
    double* __restrict__ out
) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(N) * D;
    if (idx >= total) return;
    int d = static_cast<int>(idx / static_cast<std::size_t>(N));
    int i = static_cast<int>(idx - static_cast<std::size_t>(d) * N);
    long long q = static_cast<long long>(bid[idx]) * p + static_cast<long long>(rel[idx]);
    double mn = read_le_double_dev(comp + mins_off + static_cast<std::size_t>(d) * sizeof(double));
    double x = mn + (static_cast<double>(q) + 0.5) * (2.0 * eb);
    if (output_soa) out[idx] = x;
    else out[static_cast<std::size_t>(i) * D + d] = x;
}

// -------------------------- device decode orchestration --------------------------

struct CodecBatchDeviceWorkspace {
    unsigned char* aligned_inputs = nullptr;
    unsigned char* zstd_outputs = nullptr;
    unsigned char* ans_outputs = nullptr;

    DeviceChunkCopyDesc* zstd_copy_descs = nullptr;
    void** zstd_in_ptrs = nullptr;
    std::size_t* zstd_in_bytes = nullptr;
    std::size_t* zstd_out_capacities = nullptr;
    std::size_t* zstd_actual_bytes = nullptr;
    void** zstd_out_ptrs = nullptr;
    nvcompStatus_t* zstd_statuses = nullptr;
    unsigned char* zstd_temp = nullptr;

    DeviceChunkCopyDesc* ans_copy_descs = nullptr;
    void** ans_in_ptrs = nullptr;
    std::size_t* ans_in_bytes = nullptr;
    std::size_t* ans_out_capacities = nullptr;
    std::size_t* ans_actual_bytes = nullptr;
    void** ans_out_ptrs = nullptr;
    nvcompStatus_t* ans_statuses = nullptr;
    unsigned char* ans_temp = nullptr;
};

struct CodecBatchHostMetadata {
    std::vector<DeviceChunkCopyDesc> zstd_copy_descs;
    std::vector<void*> zstd_in_ptrs;
    std::vector<std::size_t> zstd_in_bytes;
    std::vector<std::size_t> zstd_out_capacities;
    std::vector<void*> zstd_out_ptrs;

    std::vector<DeviceChunkCopyDesc> ans_copy_descs;
    std::vector<void*> ans_in_ptrs;
    std::vector<std::size_t> ans_in_bytes;
    std::vector<std::size_t> ans_out_capacities;
    std::vector<void*> ans_out_ptrs;
};

static void append_chunk_batch_metadata(
    const unsigned char* src_base,
    unsigned char* decoded_base,
    const std::vector<std::size_t>& uncomp_chunk_bytes,
    const std::vector<std::size_t>& comp_chunk_bytes,
    unsigned char* aligned_inputs,
    std::size_t& aligned_cursor,
    std::vector<DeviceChunkCopyDesc>& copy_descs,
    std::vector<void*>& in_ptrs,
    std::vector<std::size_t>& in_bytes,
    std::vector<std::size_t>& out_capacities,
    std::vector<void*>& out_ptrs
) {
    if (uncomp_chunk_bytes.size() != comp_chunk_bytes.size()) {
        throw std::runtime_error("codec chunk vectors differ in length");
    }
    std::size_t comp_cursor = 0;
    std::size_t uncomp_cursor = 0;
    for (std::size_t c = 0; c < uncomp_chunk_bytes.size(); ++c) {
        std::size_t ub = uncomp_chunk_bytes[c];
        std::size_t cb = comp_chunk_bytes[c];
        aligned_cursor = checked_align_up_size(
            aligned_cursor,
            NVCOMP_CHUNK_PTR_ALIGNMENT,
            "global aligned codec input"
        );
        unsigned char* aligned_ptr = aligned_inputs + aligned_cursor;
        DeviceChunkCopyDesc desc;
        desc.src = src_base + comp_cursor;
        desc.dst = aligned_ptr;
        desc.bytes = cb;
        copy_descs.push_back(desc);
        in_ptrs.push_back(aligned_ptr);
        in_bytes.push_back(cb);
        out_capacities.push_back(ub);
        out_ptrs.push_back(decoded_base + uncomp_cursor);
        comp_cursor = checked_add_size(comp_cursor, cb, "codec compressed cursor");
        uncomp_cursor = checked_add_size(uncomp_cursor, ub, "codec output cursor");
        aligned_cursor = checked_add_size(
            aligned_cursor, cb, "global aligned codec input"
        );
    }
}

static CodecBatchHostMetadata build_codec_batch_metadata(
    const CodecBatchPlan& plan,
    const unsigned char* d_comp,
    CodecBatchDeviceWorkspace& w
) {
    CodecBatchHostMetadata h;
    h.zstd_copy_descs.reserve(plan.zstd_num_chunks);
    h.zstd_in_ptrs.reserve(plan.zstd_num_chunks);
    h.zstd_in_bytes.reserve(plan.zstd_num_chunks);
    h.zstd_out_capacities.reserve(plan.zstd_num_chunks);
    h.zstd_out_ptrs.reserve(plan.zstd_num_chunks);

    std::size_t aligned_cursor = 0;
    for (const ZstdSpec* s : plan.zstd_specs) {
        append_chunk_batch_metadata(
            d_comp + s->data_off,
            w.zstd_outputs + s->decoded_offset,
            s->uncomp_chunk_bytes,
            s->comp_chunk_bytes,
            w.aligned_inputs,
            aligned_cursor,
            h.zstd_copy_descs,
            h.zstd_in_ptrs,
            h.zstd_in_bytes,
            h.zstd_out_capacities,
            h.zstd_out_ptrs
        );
    }
    if (h.zstd_copy_descs.size() != plan.zstd_num_chunks
        || aligned_cursor != plan.zstd_aligned_input_bytes) {
        throw std::runtime_error("global Zstd batch planning mismatch");
    }

    h.ans_copy_descs.reserve(plan.ans_num_chunks);
    h.ans_in_ptrs.reserve(plan.ans_num_chunks);
    h.ans_in_bytes.reserve(plan.ans_num_chunks);
    h.ans_out_capacities.reserve(plan.ans_num_chunks);
    h.ans_out_ptrs.reserve(plan.ans_num_chunks);

    // The aligned input arena can be reused after the Zstd call completes on
    // this same stream, so the ANS layout starts at offset zero.
    aligned_cursor = 0;
    for (const AnsBatchSpecRef& ref : plan.ans_specs) {
        if (ref.legacy != nullptr) {
            const Nva1Spec& s = *ref.legacy;
            append_chunk_batch_metadata(
                d_comp + s.data_off,
                w.ans_outputs + s.decoded_offset,
                s.uncomp_chunk_bytes,
                s.comp_chunk_bytes,
                w.aligned_inputs,
                aligned_cursor,
                h.ans_copy_descs,
                h.ans_in_ptrs,
                h.ans_in_bytes,
                h.ans_out_capacities,
                h.ans_out_ptrs
            );
        } else if (ref.wrapped != nullptr) {
            const AnsZstdSpec& s = *ref.wrapped;
            append_chunk_batch_metadata(
                w.zstd_outputs + s.zstd.decoded_offset,
                w.ans_outputs + s.decoded_offset,
                s.uncomp_chunk_bytes,
                s.comp_chunk_bytes,
                w.aligned_inputs,
                aligned_cursor,
                h.ans_copy_descs,
                h.ans_in_ptrs,
                h.ans_in_bytes,
                h.ans_out_capacities,
                h.ans_out_ptrs
            );
        } else {
            throw std::runtime_error("empty ANS batch reference");
        }
    }
    if (h.ans_copy_descs.size() != plan.ans_num_chunks
        || aligned_cursor != plan.ans_aligned_input_bytes) {
        throw std::runtime_error("global ANS batch planning mismatch");
    }
    return h;
}

template <typename T>
static void enqueue_small_h2d(
    T* dst,
    const std::vector<T>& src,
    cudaStream_t stream
) {
    if (src.empty()) return;
    CUDA_CHECK(cudaMemcpyAsync(
        dst,
        src.data(),
        src.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream
    ));
}

static void upload_codec_batch_metadata(
    const CodecBatchHostMetadata& h,
    const CodecBatchDeviceWorkspace& w,
    cudaStream_t stream
) {
    enqueue_small_h2d(w.zstd_copy_descs, h.zstd_copy_descs, stream);
    enqueue_small_h2d(w.zstd_in_ptrs, h.zstd_in_ptrs, stream);
    enqueue_small_h2d(w.zstd_in_bytes, h.zstd_in_bytes, stream);
    enqueue_small_h2d(w.zstd_out_capacities, h.zstd_out_capacities, stream);
    enqueue_small_h2d(w.zstd_out_ptrs, h.zstd_out_ptrs, stream);

    enqueue_small_h2d(w.ans_copy_descs, h.ans_copy_descs, stream);
    enqueue_small_h2d(w.ans_in_ptrs, h.ans_in_ptrs, stream);
    enqueue_small_h2d(w.ans_in_bytes, h.ans_in_bytes, stream);
    enqueue_small_h2d(w.ans_out_capacities, h.ans_out_capacities, stream);
    enqueue_small_h2d(w.ans_out_ptrs, h.ans_out_ptrs, stream);
}

static void enqueue_global_codec_batches(
    const CodecBatchPlan& plan,
    const CodecBatchDeviceWorkspace& w,
    std::size_t zstd_temp_bytes,
    std::size_t ans_temp_bytes,
    cudaStream_t stream,
    bool ans_uchar = false
) {
    constexpr int threads = 256;
    if (plan.zstd_num_chunks > 0) {
        copy_codec_chunks_to_aligned_kernel<<<
            static_cast<unsigned int>(plan.zstd_num_chunks), threads, 0, stream
        >>>(w.zstd_copy_descs, plan.zstd_num_chunks);
        CUDA_CHECK(cudaGetLastError());

        nvcompBatchedZstdDecompressOpts_t opts =
            nvcompBatchedZstdDecompressDefaultOpts;
        NVCOMP_CHECK(nvcompBatchedZstdDecompressAsync(
            reinterpret_cast<const void* const*>(w.zstd_in_ptrs),
            w.zstd_in_bytes,
            w.zstd_out_capacities,
            w.zstd_actual_bytes,
            plan.zstd_num_chunks,
            w.zstd_temp,
            zstd_temp_bytes,
            w.zstd_out_ptrs,
            opts,
            w.zstd_statuses,
            stream
        ));
    }

    if (plan.ans_num_chunks > 0) {
        // For wrapped streams, these sources are outputs of the preceding
        // Zstd batch. Stream ordering makes them ready without a host barrier.
        copy_codec_chunks_to_aligned_kernel<<<
            static_cast<unsigned int>(plan.ans_num_chunks), threads, 0, stream
        >>>(w.ans_copy_descs, plan.ans_num_chunks);
        CUDA_CHECK(cudaGetLastError());

        nvcompBatchedANSDecompressOpts_t opts =
            nvcompBatchedANSDecompressDefaultOpts;
        if (ans_uchar) opts.data_type = NVCOMP_TYPE_UCHAR;
        NVCOMP_CHECK(nvcompBatchedANSDecompressAsync(
            reinterpret_cast<const void* const*>(w.ans_in_ptrs),
            w.ans_in_bytes,
            w.ans_out_capacities,
            w.ans_actual_bytes,
            plan.ans_num_chunks,
            w.ans_temp,
            ans_temp_bytes,
            w.ans_out_ptrs,
            opts,
            w.ans_statuses,
            stream
        ));
    }
}

static void validate_batch_results_host(
    const nvcompStatus_t* statuses,
    const std::size_t* actual,
    const std::vector<std::size_t>& expected,
    const char* codec
) {
    for (std::size_t c = 0; c < expected.size(); ++c) {
        if (statuses[c] != nvcompSuccess) {
            throw std::runtime_error(
                std::string(codec) + " chunk decompression failed: "
                + nvcomp_status_string_local(statuses[c])
            );
        }
        if (actual[c] != expected[c]) {
            throw std::runtime_error(
                std::string(codec) + " decompressed chunk size mismatch"
            );
        }
    }
}

static void enqueue_i32_matrix_postprocess(
    const unsigned char* d_comp,
    const unsigned char* d_zstd_outputs,
    const unsigned char* d_ans_outputs,
    const ComponentSpec& c,
    int D,
    int n,
    int out_stride,
    int* d_out,
    int* d_tags,
    unsigned long long* d_bits,
    unsigned long long* d_offsets,
    void* d_scan_temp,
    std::size_t scan_temp_bytes,
    cudaStream_t stream
) {
    if (n <= 0 || c.kind == EncKind::EMPTY) return;
    constexpr int threads = 256;
    const int grid = (n + threads - 1) / threads;

    if (c.kind == EncKind::FIXD_MATRIX) {
        if (static_cast<int>(c.fixed_streams.size()) != D) {
            throw std::runtime_error("FIXD component D mismatch");
        }
        for (int d = 0; d < D; ++d) {
            const FixedStreamSpec& f = c.fixed_streams[static_cast<std::size_t>(d)];
            unpack_fixed_i32_kernel<<<grid, threads, 0, stream>>>(
                d_comp + f.payload_off,
                f.n,
                f.width,
                static_cast<int>(f.mn),
                d_out + static_cast<std::size_t>(d) * out_stride
            );
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (c.kind == EncKind::BANS_MATRIX) {
        if (static_cast<int>(c.bans_streams.size()) != D) {
            throw std::runtime_error("BANS component D mismatch");
        }
        for (int d = 0; d < D; ++d) {
            const BansSpec& bs = c.bans_streams[static_cast<std::size_t>(d)];
            unpack_minbytes_i32_kernel<<<grid, threads, 0, stream>>>(
                d_ans_outputs + bs.nva.decoded_offset,
                bs.n,
                bs.byte_width,
                d_out + static_cast<std::size_t>(d) * out_stride
            );
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (c.kind == EncKind::TAGGED) {
        const TaggedSpec& tg = c.tagged;
        unpack_minbytes_i32_kernel<<<grid, threads, 0, stream>>>(
            d_ans_outputs + tg.tag_bans.nva.decoded_offset,
            tg.n,
            tg.tag_bans.byte_width,
            d_tags
        );
        CUDA_CHECK(cudaGetLastError());
        compute_tag_bits_from_tags_kernel<<<grid, threads, 0, stream>>>(
            d_tags, D, n, d_bits
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            d_scan_temp, scan_temp_bytes, d_bits, d_offsets, n, stream
        ));
        const int total = D * n;
        const int grid_total = (total + threads - 1) / threads;
        unpack_tagged_values_i32_kernel<<<grid_total, threads, 0, stream>>>(
            d_comp + tg.value_payload_off,
            d_tags,
            d_offsets,
            D,
            n,
            out_stride,
            d_out
        );
        CUDA_CHECK(cudaGetLastError());
    } else if (c.kind == EncKind::ZFIXD_MATRIX) {
        const ZFixedMatrixSpec& fs = c.zfixed_matrix;
        const unsigned char* raw = d_zstd_outputs + fs.zstd.decoded_offset;
        for (int d = 0; d < D; ++d) {
            const FixedFieldSpec& f = fs.fields[static_cast<std::size_t>(d)];
            unpack_fixed_i32_kernel<<<grid, threads, 0, stream>>>(
                raw + f.payload_offset,
                f.n,
                f.width,
                static_cast<int>(f.mn),
                d_out + static_cast<std::size_t>(d) * out_stride
            );
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (c.kind == EncKind::ZANS_MATRIX) {
        const ZAnsMatrixSpec& zs = c.zans_matrix;
        const unsigned char* packed = d_ans_outputs + zs.ans.decoded_offset;
        std::size_t dimension_byte_offset = 0;
        for (int d = 0; d < D; ++d) {
            int width = zs.byte_widths[static_cast<std::size_t>(d)];
            unpack_point_major_i32_dimension_kernel<<<grid, threads, 0, stream>>>(
                packed,
                zs.n,
                width,
                zs.record_bytes,
                dimension_byte_offset,
                d_out + static_cast<std::size_t>(d) * out_stride
            );
            CUDA_CHECK(cudaGetLastError());
            dimension_byte_offset += static_cast<std::size_t>(width);
        }
    } else if (c.kind == EncKind::ZTAGGED) {
        const ZTaggedSpec& tg = c.ztagged;
        unpack_minbytes_i32_kernel<<<grid, threads, 0, stream>>>(
            d_ans_outputs + tg.tags.ans.decoded_offset,
            tg.n,
            tg.tags.byte_width,
            d_tags
        );
        CUDA_CHECK(cudaGetLastError());
        compute_tag_bits_from_tags_kernel<<<grid, threads, 0, stream>>>(
            d_tags, D, n, d_bits
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            d_scan_temp, scan_temp_bytes, d_bits, d_offsets, n, stream
        ));
        const int total = D * n;
        const int grid_total = (total + threads - 1) / threads;
        unpack_tagged_values_i32_kernel<<<grid_total, threads, 0, stream>>>(
            d_zstd_outputs + tg.value_zstd.decoded_offset,
            d_tags,
            d_offsets,
            D,
            n,
            out_stride,
            d_out
        );
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("bad integer component kind");
    }
}

static void enqueue_choice_postprocess(
    const unsigned char* d_comp,
    const unsigned char* d_zstd_outputs,
    const unsigned char* d_ans_outputs,
    const ComponentSpec& c,
    int N,
    unsigned char* d_choice,
    cudaStream_t stream
) {
    if (N <= 0) return;
    constexpr int threads = 256;
    const int grid = (N + threads - 1) / threads;

    if (c.kind == EncKind::FIXD_U8) {
        const FixedStreamSpec& f = c.fixed_streams[0];
        unpack_fixed_u8_kernel<<<grid, threads, 0, stream>>>(
            d_comp + f.payload_off, f.n, f.width, static_cast<int>(f.mn), d_choice
        );
        CUDA_CHECK(cudaGetLastError());
    } else if (c.kind == EncKind::UANS) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_choice,
            d_ans_outputs + c.uans.nva.decoded_offset,
            static_cast<std::size_t>(N),
            cudaMemcpyDeviceToDevice,
            stream
        ));
    } else if (c.kind == EncKind::ZFIXD_U8) {
        const ZFixedStreamSpec& f = c.zfixed_stream;
        unpack_fixed_u8_kernel<<<grid, threads, 0, stream>>>(
            d_zstd_outputs + f.zstd.decoded_offset,
            f.field.n,
            f.field.width,
            static_cast<int>(f.field.mn),
            d_choice
        );
        CUDA_CHECK(cudaGetLastError());
    } else if (c.kind == EncKind::ZUANS) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_choice,
            d_ans_outputs + c.zuans.ans.decoded_offset,
            static_cast<std::size_t>(N),
            cudaMemcpyDeviceToDevice,
            stream
        ));
    } else {
        throw std::runtime_error("bad choice component kind");
    }
}

static py::array_t<double> lcps_decompress_gpu_ans(py::bytes compressed, bool output_soa=false) {
    std::string s = compressed;
    std::vector<unsigned char> h_comp(s.begin(), s.end());
    HeaderSpec h = parse_lcg1(h_comp);
    CodecBatchPlan codec_plan = build_codec_batch_plan(h);

    const int N = h.N;
    const int D = h.D;
    const int threads = 256;
    const int grid_N = (N + threads - 1) / threads;
    const int total_DN = static_cast<int>(static_cast<std::size_t>(N) * D);
    const int grid_DN = (total_DN + threads - 1) / threads;
    const cudaStream_t stream = 0;

    if (codec_plan.zstd_num_chunks > std::numeric_limits<unsigned int>::max()
        || codec_plan.ans_num_chunks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("codec batch has too many chunks for a CUDA grid");
    }

    nvcompBatchedZstdDecompressOpts_t zstd_opts =
        nvcompBatchedZstdDecompressDefaultOpts;
    nvcompBatchedANSDecompressOpts_t ans_opts =
        nvcompBatchedANSDecompressDefaultOpts;
    std::size_t zstd_temp_bytes = 0;
    std::size_t ans_temp_bytes = 0;
    if (codec_plan.zstd_num_chunks > 0) {
        NVCOMP_CHECK(nvcompBatchedZstdDecompressGetTempSizeAsync(
            codec_plan.zstd_num_chunks,
            codec_plan.zstd_max_chunk,
            zstd_opts,
            &zstd_temp_bytes,
            codec_plan.zstd_total_uncomp
        ));
    }
    if (codec_plan.ans_num_chunks > 0) {
        NVCOMP_CHECK(nvcompBatchedANSDecompressGetTempSizeAsync(
            codec_plan.ans_num_chunks,
            codec_plan.ans_max_chunk,
            ans_opts,
            &ans_temp_bytes,
            codec_plan.ans_total_uncomp
        ));
    }

    // Query all CUB scratch requirements before the single device allocation.
    std::size_t choice_scan_temp_bytes = 0;
    std::size_t tag_scan_temp_bytes = 0;
    void* scan_temp_query = nullptr;
    int* null_int = nullptr;
    unsigned long long* null_ull = nullptr;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_temp_query,
        choice_scan_temp_bytes,
        null_int,
        null_int,
        N,
        stream
    ));
    if (codec_plan.max_tag_n > 0) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            scan_temp_query,
            tag_scan_temp_bytes,
            null_ull,
            null_ull,
            codec_plan.max_tag_n,
            stream
        ));
    }
    // CUB's execution API takes temp_storage_bytes as a non-const size_t&.
    // Keep this mutable even though the allocated capacity is fixed below.
    std::size_t scan_temp_bytes = std::max(
        choice_scan_temp_bytes, tag_scan_temp_bytes
    );

    const std::size_t zstd_chunks_alloc = std::max<std::size_t>(
        1, codec_plan.zstd_num_chunks
    );
    const std::size_t ans_chunks_alloc = std::max<std::size_t>(
        1, codec_plan.ans_num_chunks
    );
    const std::size_t aligned_input_bytes = std::max<std::size_t>(
        1,
        std::max(
            codec_plan.zstd_aligned_input_bytes,
            codec_plan.ans_aligned_input_bytes
        )
    );

    // Lay out every device allocation up front. DeviceBuffer below performs
    // the only explicit cudaMalloc in the active decompression path.
    ArenaLayout layout;
    const std::size_t off_comp = layout.reserve_bytes(
        h_comp.size(), NVCOMP_CHUNK_PTR_ALIGNMENT, "compressed container"
    );
    const std::size_t off_zstd_outputs = layout.reserve_bytes(
        std::max<std::size_t>(1, codec_plan.zstd_output_bytes),
        NVCOMP_CHUNK_PTR_ALIGNMENT,
        "Zstd output arena"
    );
    const std::size_t off_ans_outputs = layout.reserve_bytes(
        std::max<std::size_t>(1, codec_plan.ans_output_bytes),
        NVCOMP_CHUNK_PTR_ALIGNMENT,
        "ANS output arena"
    );
    const std::size_t off_aligned_inputs = layout.reserve_bytes(
        aligned_input_bytes,
        NVCOMP_CHUNK_PTR_ALIGNMENT,
        "aligned codec inputs"
    );
    const std::size_t off_zstd_temp = layout.reserve_bytes(
        std::max<std::size_t>(1, zstd_temp_bytes),
        NVCOMP_CHUNK_PTR_ALIGNMENT,
        "Zstd temp"
    );
    const std::size_t off_ans_temp = layout.reserve_bytes(
        std::max<std::size_t>(1, ans_temp_bytes),
        NVCOMP_CHUNK_PTR_ALIGNMENT,
        "ANS temp"
    );

    const std::size_t off_zstd_copy_descs = layout.reserve<DeviceChunkCopyDesc>(
        zstd_chunks_alloc, "Zstd copy descriptors"
    );
    const std::size_t off_zstd_in_ptrs = layout.reserve<void*>(
        zstd_chunks_alloc, "Zstd input pointers"
    );
    const std::size_t off_zstd_in_bytes = layout.reserve<std::size_t>(
        zstd_chunks_alloc, "Zstd input sizes"
    );
    const std::size_t off_zstd_out_capacities = layout.reserve<std::size_t>(
        zstd_chunks_alloc, "Zstd output capacities"
    );
    const std::size_t off_zstd_actual_bytes = layout.reserve<std::size_t>(
        zstd_chunks_alloc, "Zstd actual sizes"
    );
    const std::size_t off_zstd_out_ptrs = layout.reserve<void*>(
        zstd_chunks_alloc, "Zstd output pointers"
    );
    const std::size_t off_zstd_statuses = layout.reserve<nvcompStatus_t>(
        zstd_chunks_alloc, "Zstd statuses"
    );

    const std::size_t off_ans_copy_descs = layout.reserve<DeviceChunkCopyDesc>(
        ans_chunks_alloc, "ANS copy descriptors"
    );
    const std::size_t off_ans_in_ptrs = layout.reserve<void*>(
        ans_chunks_alloc, "ANS input pointers"
    );
    const std::size_t off_ans_in_bytes = layout.reserve<std::size_t>(
        ans_chunks_alloc, "ANS input sizes"
    );
    const std::size_t off_ans_out_capacities = layout.reserve<std::size_t>(
        ans_chunks_alloc, "ANS output capacities"
    );
    const std::size_t off_ans_actual_bytes = layout.reserve<std::size_t>(
        ans_chunks_alloc, "ANS actual sizes"
    );
    const std::size_t off_ans_out_ptrs = layout.reserve<void*>(
        ans_chunks_alloc, "ANS output pointers"
    );
    const std::size_t off_ans_statuses = layout.reserve<nvcompStatus_t>(
        ans_chunks_alloc, "ANS statuses"
    );

    const std::size_t out_block_count = checked_mul_size(
        static_cast<std::size_t>(D),
        static_cast<std::size_t>(std::max(1, h.num_block)),
        "outBlock"
    );
    const std::size_t dn_count = checked_mul_size(
        static_cast<std::size_t>(D), static_cast<std::size_t>(N), "D*N"
    );
    const std::size_t tag_count = static_cast<std::size_t>(
        std::max(1, codec_plan.max_tag_n)
    );

    const std::size_t off_out_block = layout.reserve<int>(out_block_count, "outBlock");
    const std::size_t off_idrel = layout.reserve<int>(
        static_cast<std::size_t>(std::max(1, h.num_idrel)), "IDRel"
    );
    const std::size_t off_rel = layout.reserve<int>(dn_count, "relative offsets");
    const std::size_t off_choice = layout.reserve<unsigned char>(N, "choice");
    const std::size_t off_tags = layout.reserve<int>(tag_count, "tag scratch");
    const std::size_t off_tag_bits = layout.reserve<unsigned long long>(
        tag_count, "tag bit counts"
    );
    const std::size_t off_tag_offsets = layout.reserve<unsigned long long>(
        tag_count, "tag bit offsets"
    );
    const std::size_t off_scan_temp = layout.reserve_bytes(
        std::max<std::size_t>(1, scan_temp_bytes), 256, "CUB scan temp"
    );
    const std::size_t off_flag_idrel = layout.reserve<int>(N, "IDRel flags");
    const std::size_t off_flag_block = layout.reserve<int>(N, "block flags");
    const std::size_t off_pos_idrel = layout.reserve<int>(N, "IDRel positions");
    const std::size_t off_pos_block = layout.reserve<int>(N, "block positions");
    const std::size_t off_bid = layout.reserve<int>(dn_count, "block IDs");
    const std::size_t off_parent = layout.reserve<int>(N, "parents");
    const std::size_t off_parent_tmp = layout.reserve<int>(N, "temporary parents");
    const std::size_t off_out = layout.reserve<double>(dn_count, "coordinates");

    DeviceBuffer<unsigned char> d_arena(std::max<std::size_t>(1, layout.bytes));
    unsigned char* const arena = d_arena.ptr;

    unsigned char* const d_comp = arena_ptr<unsigned char>(arena, off_comp);
    unsigned char* const d_zstd_outputs = arena_ptr<unsigned char>(arena, off_zstd_outputs);
    unsigned char* const d_ans_outputs = arena_ptr<unsigned char>(arena, off_ans_outputs);
    int* const d_outBlock = arena_ptr<int>(arena, off_out_block);
    int* const d_IDRel = arena_ptr<int>(arena, off_idrel);
    int* const d_rel = arena_ptr<int>(arena, off_rel);
    unsigned char* const d_choice = arena_ptr<unsigned char>(arena, off_choice);
    int* const d_tags = arena_ptr<int>(arena, off_tags);
    unsigned long long* const d_tag_bits = arena_ptr<unsigned long long>(arena, off_tag_bits);
    unsigned long long* const d_tag_offsets = arena_ptr<unsigned long long>(arena, off_tag_offsets);
    void* const d_scan_temp = arena_ptr<unsigned char>(arena, off_scan_temp);
    int* const d_flag_idrel = arena_ptr<int>(arena, off_flag_idrel);
    int* const d_flag_block = arena_ptr<int>(arena, off_flag_block);
    int* const d_pos_idrel = arena_ptr<int>(arena, off_pos_idrel);
    int* const d_pos_block = arena_ptr<int>(arena, off_pos_block);
    int* const d_bid = arena_ptr<int>(arena, off_bid);
    int* const d_parent = arena_ptr<int>(arena, off_parent);
    int* const d_parent_tmp = arena_ptr<int>(arena, off_parent_tmp);
    double* const d_out = arena_ptr<double>(arena, off_out);

    CodecBatchDeviceWorkspace codec_workspace;
    codec_workspace.aligned_inputs = arena_ptr<unsigned char>(arena, off_aligned_inputs);
    codec_workspace.zstd_outputs = d_zstd_outputs;
    codec_workspace.ans_outputs = d_ans_outputs;
    codec_workspace.zstd_copy_descs = arena_ptr<DeviceChunkCopyDesc>(arena, off_zstd_copy_descs);
    codec_workspace.zstd_in_ptrs = arena_ptr<void*>(arena, off_zstd_in_ptrs);
    codec_workspace.zstd_in_bytes = arena_ptr<std::size_t>(arena, off_zstd_in_bytes);
    codec_workspace.zstd_out_capacities = arena_ptr<std::size_t>(arena, off_zstd_out_capacities);
    codec_workspace.zstd_actual_bytes = arena_ptr<std::size_t>(arena, off_zstd_actual_bytes);
    codec_workspace.zstd_out_ptrs = arena_ptr<void*>(arena, off_zstd_out_ptrs);
    codec_workspace.zstd_statuses = arena_ptr<nvcompStatus_t>(arena, off_zstd_statuses);
    codec_workspace.zstd_temp = arena_ptr<unsigned char>(arena, off_zstd_temp);
    codec_workspace.ans_copy_descs = arena_ptr<DeviceChunkCopyDesc>(arena, off_ans_copy_descs);
    codec_workspace.ans_in_ptrs = arena_ptr<void*>(arena, off_ans_in_ptrs);
    codec_workspace.ans_in_bytes = arena_ptr<std::size_t>(arena, off_ans_in_bytes);
    codec_workspace.ans_out_capacities = arena_ptr<std::size_t>(arena, off_ans_out_capacities);
    codec_workspace.ans_actual_bytes = arena_ptr<std::size_t>(arena, off_ans_actual_bytes);
    codec_workspace.ans_out_ptrs = arena_ptr<void*>(arena, off_ans_out_ptrs);
    codec_workspace.ans_statuses = arena_ptr<nvcompStatus_t>(arena, off_ans_statuses);
    codec_workspace.ans_temp = arena_ptr<unsigned char>(arena, off_ans_temp);

    CodecBatchHostMetadata codec_metadata = build_codec_batch_metadata(
        codec_plan, d_comp, codec_workspace
    );

    // Pinned destinations keep the final D2H and small status copies truly
    // asynchronous until the one explicit synchronization below.
    ArenaLayout host_layout;
    const std::size_t host_off_zstd_statuses = host_layout.reserve<nvcompStatus_t>(
        zstd_chunks_alloc, "host Zstd statuses"
    );
    const std::size_t host_off_zstd_actual = host_layout.reserve<std::size_t>(
        zstd_chunks_alloc, "host Zstd actual sizes"
    );
    const std::size_t host_off_ans_statuses = host_layout.reserve<nvcompStatus_t>(
        ans_chunks_alloc, "host ANS statuses"
    );
    const std::size_t host_off_ans_actual = host_layout.reserve<std::size_t>(
        ans_chunks_alloc, "host ANS actual sizes"
    );
    const std::size_t host_off_output = host_layout.reserve<double>(
        dn_count, "pinned coordinate output"
    );
    PinnedHostBuffer<unsigned char> h_result_arena(
        std::max<std::size_t>(1, host_layout.bytes)
    );
    unsigned char* const host_arena = h_result_arena.ptr;
    nvcompStatus_t* const h_zstd_statuses = arena_ptr<nvcompStatus_t>(
        host_arena, host_off_zstd_statuses
    );
    std::size_t* const h_zstd_actual = arena_ptr<std::size_t>(
        host_arena, host_off_zstd_actual
    );
    nvcompStatus_t* const h_ans_statuses = arena_ptr<nvcompStatus_t>(
        host_arena, host_off_ans_statuses
    );
    std::size_t* const h_ans_actual = arena_ptr<std::size_t>(
        host_arena, host_off_ans_actual
    );
    double* const h_output = arena_ptr<double>(host_arena, host_off_output);

    std::vector<py::ssize_t> shape;
    if (output_soa) {
        shape = {static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(N)};
    } else {
        shape = {static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(D)};
    }
    py::array_t<double> arr(shape);
    py::buffer_info info = arr.request();

    // All explicit device allocations are complete. This is the one bulk H2D.
    CUDA_CHECK(cudaMemcpyAsync(
        d_comp,
        h_comp.data(),
        h_comp.size(),
        cudaMemcpyHostToDevice,
        stream
    ));
    // Small pointer/size/descriptor uploads are intentionally separate.
    upload_codec_batch_metadata(codec_metadata, codec_workspace, stream);

    enqueue_global_codec_batches(
        codec_plan,
        codec_workspace,
        zstd_temp_bytes,
        ans_temp_bytes,
        stream
    );

    // Decode the four logical components from the already-batched codec
    // outputs. Everything stays ordered on the same stream without barriers.
    enqueue_i32_matrix_postprocess(
        d_comp, d_zstd_outputs, d_ans_outputs,
        h.blk, D, h.num_block, std::max(1, h.num_block), d_outBlock,
        d_tags, d_tag_bits, d_tag_offsets, d_scan_temp, scan_temp_bytes, stream
    );
    enqueue_i32_matrix_postprocess(
        d_comp, d_zstd_outputs, d_ans_outputs,
        h.idr, 1, h.num_idrel, std::max(1, h.num_idrel), d_IDRel,
        d_tags, d_tag_bits, d_tag_offsets, d_scan_temp, scan_temp_bytes, stream
    );
    enqueue_i32_matrix_postprocess(
        d_comp, d_zstd_outputs, d_ans_outputs,
        h.rel, D, N, N, d_rel,
        d_tags, d_tag_bits, d_tag_offsets, d_scan_temp, scan_temp_bytes, stream
    );
    enqueue_choice_postprocess(
        d_comp, d_zstd_outputs, d_ans_outputs, h.cho, N, d_choice, stream
    );

    // Reconstruct bid using choice and IDRel/outBlock.
    flags_from_choice_kernel<<<grid_N, threads, 0, stream>>>(
        d_choice, N, d_flag_idrel, d_flag_block
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        d_scan_temp,
        scan_temp_bytes,
        d_flag_idrel,
        d_pos_idrel,
        N,
        stream
    ));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        d_scan_temp,
        scan_temp_bytes,
        d_flag_block,
        d_pos_block,
        N,
        stream
    ));

    init_bid_and_parent_kernel<<<grid_N, threads, 0, stream>>>(
        d_choice,
        d_pos_idrel,
        d_pos_block,
        d_IDRel,
        d_outBlock,
        N,
        D,
        std::max(1, h.num_block),
        d_bid,
        d_parent
    );
    CUDA_CHECK(cudaGetLastError());

    int iters = 0;
    while ((1u << iters) < static_cast<unsigned>(N) && iters < 31) ++iters;
    int* parent_in = d_parent;
    int* parent_out = d_parent_tmp;
    for (int it = 0; it < iters; ++it) {
        pointer_jump_kernel<<<grid_N, threads, 0, stream>>>(
            parent_in, parent_out, N
        );
        CUDA_CHECK(cudaGetLastError());
        std::swap(parent_in, parent_out);
    }

    fill_bid_from_root_kernel<<<grid_DN, threads, 0, stream>>>(
        d_bid, parent_in, N, D
    );
    CUDA_CHECK(cudaGetLastError());
    reconstruct_coords_kernel<<<grid_DN, threads, 0, stream>>>(
        d_bid,
        d_rel,
        d_comp,
        h.mins_off,
        N,
        D,
        h.p,
        h.eb,
        output_soa,
        d_out
    );
    CUDA_CHECK(cudaGetLastError());

    // Defer all small status transfers and the only bulk D2H until the end.
    if (codec_plan.zstd_num_chunks > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            h_zstd_statuses,
            codec_workspace.zstd_statuses,
            codec_plan.zstd_num_chunks * sizeof(nvcompStatus_t),
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_CHECK(cudaMemcpyAsync(
            h_zstd_actual,
            codec_workspace.zstd_actual_bytes,
            codec_plan.zstd_num_chunks * sizeof(std::size_t),
            cudaMemcpyDeviceToHost,
            stream
        ));
    }
    if (codec_plan.ans_num_chunks > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            h_ans_statuses,
            codec_workspace.ans_statuses,
            codec_plan.ans_num_chunks * sizeof(nvcompStatus_t),
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_CHECK(cudaMemcpyAsync(
            h_ans_actual,
            codec_workspace.ans_actual_bytes,
            codec_plan.ans_num_chunks * sizeof(std::size_t),
            cudaMemcpyDeviceToHost,
            stream
        ));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        h_output,
        d_out,
        dn_count * sizeof(double),
        cudaMemcpyDeviceToHost,
        stream
    ));

    // The only explicit synchronization in the active path. d_arena is freed
    // by RAII only after this final D2H and validation complete.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    validate_batch_results_host(
        h_zstd_statuses,
        h_zstd_actual,
        codec_metadata.zstd_out_capacities,
        "Zstd"
    );
    validate_batch_results_host(
        h_ans_statuses,
        h_ans_actual,
        codec_metadata.ans_out_capacities,
        "ANS"
    );

    std::memcpy(info.ptr, h_output, dn_count * sizeof(double));

    return arr;
}

// LCG2 extends the existing host-plan -> single arena -> global codec batches
// -> GPU reconstruction -> final D2H architecture. LCG1 above is retained.
namespace lcg2 {

enum class Encoding : unsigned char { Empty=0, Tagged=1, Fixed=2, Raw=3 };
struct Component {
    Encoding kind = Encoding::Empty;
    int dims = 0, n = 0;
    unsigned long long tagged_bits = 0;
    std::vector<FixedFieldSpec> fields;
    Nva1Spec ans;
    ZstdSpec zstd;
};
struct Header {
    int dims=0, n=0, p=0, wrap_min=0, wrap_max=0;
    float eb=0;
    std::size_t minima_off=0;
    std::array<Component,5> components;
};
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(std::string("LCG2: ")+message);
}
static std::size_t mul(std::size_t a, std::size_t b) {
    return checked_mul_size(a,b,"LCG2 size");
}
static std::size_t add(std::size_t a, std::size_t b) {
    return checked_add_size(a,b,"LCG2 size");
}
static std::size_t bytes_for_bits(std::size_t bits) { return bits/8+(bits%8!=0); }
static int grid_for(std::size_t n) {
    return static_cast<int>(std::max<std::size_t>(1,std::min<std::size_t>(65535,n/256+(n%256!=0))));
}

// Parse the section tables on the host; actual tags/values stay on the GPU.
// Reuse Nva1Spec/ZstdSpec as chunk descriptors, without requiring legacy magic.
static ChunkedCodecSpec parse_section(Reader& r, std::size_t base,
                                      int expected_codec, std::size_t expected_raw) {
    ChunkedCodecSpec s;
    s.off=add(base,r.pos);
    require(r.read_u8()==expected_codec,"unexpected section codec");
    s.total_uncomp=u64_to_size(r.read_le<std::uint64_t>(),"LCG2 raw section");
    require(s.total_uncomp==expected_raw && expected_raw>0,"section raw length mismatch");
    s.num_chunks=r.read_le<std::uint32_t>();
    const std::size_t limit=expected_codec==1 ? (1u<<24) : 65536;
    const std::size_t expected_chunks=expected_raw/limit+(expected_raw%limit!=0);
    require(s.num_chunks==expected_chunks,"section chunk count mismatch");
    r.require(mul(s.num_chunks,16));
    s.uncomp_chunk_bytes.reserve(s.num_chunks);
    s.comp_chunk_bytes.reserve(s.num_chunks);
    std::size_t remaining=expected_raw;
    for (std::uint32_t i=0;i<s.num_chunks;++i) {
        const auto ub=u64_to_size(r.read_le<std::uint64_t>(),"LCG2 raw chunk");
        const auto cb=u64_to_size(r.read_le<std::uint64_t>(),"LCG2 compressed chunk");
        require(ub==std::min(limit,remaining) && cb>0,"invalid chunk lengths");
        s.uncomp_chunk_bytes.push_back(ub); s.comp_chunk_bytes.push_back(cb);
        remaining-=ub; s.max_chunk=std::max(s.max_chunk,ub);
        s.comp_total=add(s.comp_total,cb);
    }
    require(remaining==0,"incomplete section");
    s.data_off=add(base,r.pos);
    r.skip(s.comp_total);
    s.len=add(base,r.pos)-s.off;
    return s;
}

static Component parse_component(Reader& top, const char* name, int dims, int max_n) {
    top.expect_magic(name);
    const auto length=u64_to_size(top.read_le<std::uint64_t>(),"LCG2 component");
    top.require(length);
    const auto base=top.pos;
    Reader r(top.data+base,length);
    Component c;
    const auto kind=r.read_u8();
    require(kind<=3,"unknown encoding"); c.kind=static_cast<Encoding>(kind);
    c.dims=u64_to_int(r.read_le<std::uint32_t>(),"LCG2 dimensions");
    c.n=u64_to_int(r.read_le<std::uint64_t>(),"LCG2 records");
    require(c.dims==dims && c.n<=max_n,"component shape mismatch");
    const auto values=mul(c.n,dims);
    std::size_t fixed_bytes=0;
    if (c.kind==Encoding::Fixed) {
        r.require(mul(dims,13)); c.fields.resize(dims);
        for (auto& f:c.fields) {
            f.mn=r.read_le<std::int32_t>(); f.width=r.read_u8(); f.n=c.n;
            f.payload_bytes=u64_to_size(r.read_le<std::uint64_t>(),"LCG2 fixed bytes");
            require(f.mn>=0 && f.width>=1 && f.width<=31,"invalid fixed field");
            require(f.payload_bytes==bytes_for_bits(mul(c.n,f.width)),"fixed length mismatch");
            f.payload_offset=fixed_bytes; fixed_bytes=add(fixed_bytes,f.payload_bytes);
        }
    } else if (c.kind==Encoding::Tagged) {
        c.tagged_bits=r.read_le<std::uint64_t>();
        require(c.tagged_bits>=values && c.tagged_bits<=mul(values,31),"invalid packed bit count");
    }
    const int sections=r.read_u8();
    if (c.kind==Encoding::Empty) {
        require(c.n==0 && sections==0,"invalid empty component");
    } else {
        require(c.n>0,"nonempty encoding with no records");
        if (c.kind==Encoding::Tagged) {
            require(sections==2,"Tagged needs ANS tags and Zstd values");
            static_cast<ChunkedCodecSpec&>(c.ans)=parse_section(r,base,1,c.n);
            static_cast<ChunkedCodecSpec&>(c.zstd)=parse_section(
                r,base,2,bytes_for_bits(u64_to_size(c.tagged_bits,"LCG2 tagged bits")));
        } else if (c.kind==Encoding::Fixed) {
            require(sections==1,"Fixed needs one Zstd section");
            static_cast<ChunkedCodecSpec&>(c.zstd)=parse_section(r,base,2,fixed_bytes);
        } else {
            require(sections==1,"Raw needs one ANS section");
            static_cast<ChunkedCodecSpec&>(c.ans)=parse_section(r,base,1,mul(values,4));
        }
    }
    require(r.remaining()==0,"trailing component bytes");
    top.skip(length);
    return c;
}

static Header parse(const std::vector<unsigned char>& data) {
    Reader r(data.data(),data.size()); r.expect_magic("LCG2");
    require(r.read_le<std::uint32_t>()==1,"unsupported version");
    Header h;
    h.dims=u64_to_int(r.read_le<std::uint32_t>(),"LCG2 D");
    h.n=u64_to_int(r.read_le<std::uint64_t>(),"LCG2 N");
    h.p=r.read_le<std::int32_t>(); h.eb=r.read_le<float>();
    h.wrap_min=r.read_le<std::int32_t>(); h.wrap_max=r.read_le<std::int32_t>();
    require(h.dims>0 && h.n>0 && h.p>0,"invalid shape or block size");
    require(h.eb>0 && std::isfinite(2.0f*h.eb),"invalid error bound");
    require(h.wrap_min>=0 && h.wrap_max>=h.wrap_min,"invalid wrap range");
    const auto values=mul(h.n,h.dims);
    require(mul(values,sizeof(float))<=static_cast<std::size_t>(PY_SSIZE_T_MAX),"output too large");
    h.minima_off=r.pos;
    r.require(mul(h.dims,sizeof(float)));
    for (int d=0;d<h.dims;++d) require(std::isfinite(r.read_le<float>()),"nonfinite minimum");
    require(r.read_le<std::uint32_t>()==5,"expected five components");
    const char* names[]={"IDR0","DIF0","KEP0","CH20","REL0"};
    for (int s=0;s<5;++s)
        h.components[s]=parse_component(r,names[s],s==0 || s==3 ? 1:h.dims,h.n);
    require(r.remaining()==0,"trailing container bytes");
    const auto& c=h.components;
    require(c[0].n==h.n && c[4].n==h.n,"coordinate stream count mismatch");
    require(c[3].n>0 && c[2].n>0
            && std::size_t(c[1].n)+c[2].n==std::size_t(c[3].n),"diff/keep count mismatch");
    return h;
}

static CodecBatchPlan make_plan(Header& h) {
    CodecBatchPlan plan;
    for (auto& c:h.components) {
        if (c.kind==Encoding::Tagged || c.kind==Encoding::Raw) plan_legacy_ans_spec(c.ans,plan);
        if (c.kind==Encoding::Tagged || c.kind==Encoding::Fixed) plan_zstd_spec(c.zstd,plan);
        if (c.kind==Encoding::Tagged) plan.max_tag_n=std::max(plan.max_tag_n,c.n);
    }
    return plan;
}

__global__ void unpack_raw_kernel(const unsigned char* raw, int n, int dims, int* out, int* error) {
    const std::size_t total=std::size_t(n)*dims;
    for (std::size_t j=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         j<total;j+=std::size_t(blockDim.x)*gridDim.x) {
        const auto d=j/n, i=j%n, at=(i*dims+d)*4;
        std::uint32_t v=0;
        for (int b=0;b<4;++b) v|=std::uint32_t(raw[at+b])<<(8*b);
        if (v>INT_MAX) { atomicOr(error,1); v=0; }
        out[j]=static_cast<int>(v);
    }
}
__global__ void unpack_fixed_kernel(const unsigned char* raw, int n, int width,
                                    int minimum, int* out, int* error) {
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i<std::size_t(n);i+=std::size_t(blockDim.x)*gridDim.x) {
        const auto begin=i*width; std::uint32_t v=0;
        for (int b=0;b<width;++b) {
            const auto bit=begin+b;
            v=(v<<1)|((raw[bit/8]>>(7-bit%8))&1u);
        }
        v+=static_cast<std::uint32_t>(minimum);
        if (v>INT_MAX) { atomicOr(error,1); v=0; }
        out[i]=static_cast<int>(v);
    }
}
__global__ void tag_bits_kernel(const unsigned char* tags, int n, int dims,
                                unsigned long long* bits, int* error) {
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i<std::size_t(n);i+=std::size_t(blockDim.x)*gridDim.x) {
        const unsigned tag=tags[i];
        if (tag<1 || tag>31) { atomicOr(error,2); bits[i]=0; }
        else bits[i]=static_cast<unsigned long long>(tag)*dims;
    }
}
__global__ void unpack_tagged_kernel(const unsigned char* raw, const unsigned char* tags,
    const unsigned long long* offsets, unsigned long long valid_bits, int n, int dims,
    int* out, int* error) {
    const std::size_t total=std::size_t(n)*dims;
    for (std::size_t j=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         j<total;j+=std::size_t(blockDim.x)*gridDim.x) {
        const auto d=j/n,i=j%n;
        const unsigned tag=tags[i];
        const auto record_bits=static_cast<unsigned long long>(tag)*dims;
        if (tag<1 || tag>31 || offsets[i]>valid_bits || record_bits>valid_bits-offsets[i]) {
            atomicOr(error,2); out[j]=0; continue;
        }
        if (i+1==std::size_t(n) && offsets[i]+record_bits!=valid_bits) atomicOr(error,2);
        const auto begin=offsets[i]+d*tag;
        std::uint32_t v=0;
        for (unsigned b=0;b<tag;++b) {
            const auto bit=begin+b;
            v=(v<<1)|((raw[bit/8]>>(7-bit%8))&1u);
        }
        out[j]=static_cast<int>(v);
    }
}

static void unpack_component(const Component& c, const CodecBatchDeviceWorkspace& w,
    int* out, unsigned long long* bits, unsigned long long* offsets,
    void* temp, std::size_t capacity, int* error, cudaStream_t stream) {
    if (c.n==0) return;
    const int grid=grid_for(c.n);
    if (c.kind==Encoding::Raw) {
        unpack_raw_kernel<<<grid_for(mul(c.n,c.dims)),256,0,stream>>>(
            w.ans_outputs+c.ans.decoded_offset,c.n,c.dims,out,error);
        CUDA_CHECK(cudaGetLastError());
    } else if (c.kind==Encoding::Fixed) {
        for (int d=0;d<c.dims;++d) {
            const auto& f=c.fields[d];
            unpack_fixed_kernel<<<grid,256,0,stream>>>(
                w.zstd_outputs+c.zstd.decoded_offset+f.payload_offset,
                c.n,f.width,static_cast<int>(f.mn),out+std::size_t(d)*c.n,error);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        const auto* tags=w.ans_outputs+c.ans.decoded_offset;
        tag_bits_kernel<<<grid,256,0,stream>>>(tags,c.n,c.dims,bits,error);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp,capacity,bits,offsets,c.n,stream));
        unpack_tagged_kernel<<<grid_for(mul(c.n,c.dims)),256,0,stream>>>(
            w.zstd_outputs+c.zstd.decoded_offset,tags,offsets,c.tagged_bits,c.n,c.dims,out,error);
        CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void flags_kernel(const int* values, int n, bool choice2, int* flags, int* error) {
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i<std::size_t(n);i+=std::size_t(blockDim.x)*gridDim.x) {
        const int v=values[i];
        if (choice2) {
            if (v<0 || v>1 || (i==0 && v!=0)) atomicOr(error,4);
            flags[i]=v==1;
        } else {
            if (v<0 || std::size_t(v)>=std::size_t(n)-i) atomicOr(error,8);
            flags[i]=v==0;
        }
    }
}
__global__ void check_count_kernel(const int* flags, const int* positions,
                                   int n, int expected, int* error) {
    if (threadIdx.x==0 && blockIdx.x==0
        && positions[n-1]+flags[n-1]!=expected) atomicOr(error,16);
}

// Each block record is either a constant reset (keep) or a modular increment
// (-inverse_zigzag(diff)). Composition is associative, so CUB performs a
// parallel segmented prefix scan even for one very long run of differences.
struct ResetValue { std::uint32_t value, reset; };
struct ModularResetAdd {
    std::uint32_t range;
    __host__ __device__ ResetValue operator()(ResetValue left, ResetValue right) const {
        if (right.reset) return right;
        auto sum=left.value+right.value; // <=2*(range-1), fits uint32
        if (sum>=range) sum-=range;
        return {sum,left.reset};
    }
};
__global__ void block_terms_kernel(const int* choices, const int* diff_positions,
    const int* differences, const int* kept, int blocks, int ndiff, int nkeep,
    int minimum, std::uint32_t range, ResetValue* terms, int* error) {
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i<std::size_t(blocks);i+=std::size_t(blockDim.x)*gridDim.x) {
        const int rank=diff_positions[i];
        const bool difference=choices[i]==1;
        ResetValue term{0,1};
        if (difference) {
            if (rank<0 || rank>=ndiff) { atomicOr(error,16); terms[i]=term; continue; }
            const int symbol=differences[rank];
            if (symbol<0 || std::uint32_t(symbol)>=range) { atomicOr(error,32); terms[i]=term; continue; }
            const int correction=(symbol&1) ? -(symbol/2)-1 : symbol/2;
            term.value=correction>0 ? range-std::uint32_t(correction) : std::uint32_t(-correction);
            term.reset=0;
        } else {
            const int pos=static_cast<int>(i)-rank;
            if (pos<0 || pos>=nkeep) { atomicOr(error,16); terms[i]=term; continue; }
            const int value=kept[pos];
            if (value<minimum || std::uint32_t(value-minimum)>=range) atomicOr(error,32);
            else term.value=static_cast<std::uint32_t>(value-minimum);
        }
        terms[i]=term;
    }
}
__global__ void store_blocks_kernel(const ResetValue* prefix, int n, int minimum, int* blocks) {
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i<std::size_t(n);i+=std::size_t(blockDim.x)*gridDim.x)
        blocks[i]=minimum+static_cast<int>(prefix[i].value);
}
__global__ void init_chain_kernel(const int* idrel, const int* positions, const int* blocks,
    int n, int dims, int count, int* bid, int* parents, int* error) {
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i<std::size_t(n);i+=std::size_t(blockDim.x)*gridDim.x) {
        const int delta=idrel[i];
        parents[i]=static_cast<int>(i);
        for (int d=0;d<dims;++d) bid[std::size_t(d)*n+i]=0;
        if (delta>0 && std::size_t(delta)<std::size_t(n)-i) parents[i]=static_cast<int>(i)+delta;
        else if (delta==0) {
            const int pos=positions[i];
            if (pos<0 || pos>=count) { atomicOr(error,16); continue; }
            for (int d=0;d<dims;++d) bid[std::size_t(d)*n+i]=blocks[std::size_t(d)*count+pos];
        } else atomicOr(error,8);
    }
}
__global__ void reconstruct_kernel(const int* bid, const int* root, const int* rel,
    const unsigned char* container, std::size_t minima_off, int n, int dims,
    int p, float eb, bool soa, float* output, int* error) {
    const std::size_t total=std::size_t(n)*dims;
    for (std::size_t j=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         j<total;j+=std::size_t(blockDim.x)*gridDim.x) {
        const auto d=j/n,i=j%n;
        const int block=bid[d*n+root[i]], offset=rel[j];
        const long long q64=static_cast<long long>(block)*p+offset;
        const auto dest=soa ? j : i*dims+d;
        if (block<0 || offset<0 || offset>=p || q64<0 || q64>INT_MAX) {
            atomicOr(error,64); output[dest]=0; continue;
        }
        const unsigned char* source=container+minima_off+d*4;
        std::uint32_t u=0;
        for (int b=0;b<4;++b) u|=std::uint32_t(source[b])<<(8*b);
        float minimum; memcpy(&minimum,&u,4);
        const float midpoint=__fadd_rn(__int2float_rn(static_cast<int>(q64)),0.5f);
        output[dest]=__fmaf_rn(midpoint,__fmul_rn(2.0f,eb),minimum);
        if (!isfinite(output[dest])) atomicOr(error,64);
    }
}

struct BatchOffsets {
    std::size_t copies,in_ptrs,in_bytes,capacities,actual,out_ptrs,statuses,temp;
    BatchOffsets(ArenaLayout& layout,std::size_t count,std::size_t scratch) {
        count=std::max<std::size_t>(1,count);
        copies=layout.reserve<DeviceChunkCopyDesc>(count,"LCG2 chunk copies");
        in_ptrs=layout.reserve<void*>(count,"LCG2 chunk pointers");
        in_bytes=layout.reserve<std::size_t>(count,"LCG2 chunk lengths");
        capacities=layout.reserve<std::size_t>(count,"LCG2 chunk capacities");
        actual=layout.reserve<std::size_t>(count,"LCG2 actual lengths");
        out_ptrs=layout.reserve<void*>(count,"LCG2 output pointers");
        statuses=layout.reserve<nvcompStatus_t>(count,"LCG2 codec statuses");
        temp=layout.reserve_bytes(std::max<std::size_t>(1,scratch),256,"LCG2 codec scratch");
    }
};
struct CodecOffsets {
    std::size_t zstd_output,ans_output,aligned_input;
    BatchOffsets zstd,ans;
    CodecOffsets(ArenaLayout& layout,const CodecBatchPlan& plan,std::size_t ztemp,std::size_t atemp)
        : zstd(layout,plan.zstd_num_chunks,ztemp),ans(layout,plan.ans_num_chunks,atemp) {
        zstd_output=layout.reserve_bytes(std::max<std::size_t>(1,plan.zstd_output_bytes),256,"LCG2 Zstd arena");
        ans_output=layout.reserve_bytes(std::max<std::size_t>(1,plan.ans_output_bytes),256,"LCG2 ANS arena");
        aligned_input=layout.reserve_bytes(std::max({std::size_t(1),plan.zstd_aligned_input_bytes,
                                                   plan.ans_aligned_input_bytes}),256,"LCG2 aligned inputs");
    }
    CodecBatchDeviceWorkspace bind(unsigned char* base) const {
        CodecBatchDeviceWorkspace w;
        w.zstd_outputs=base+zstd_output; w.ans_outputs=base+ans_output;
        w.aligned_inputs=base+aligned_input;
        w.zstd_copy_descs=arena_ptr<DeviceChunkCopyDesc>(base,zstd.copies);
        w.zstd_in_ptrs=arena_ptr<void*>(base,zstd.in_ptrs);
        w.zstd_in_bytes=arena_ptr<std::size_t>(base,zstd.in_bytes);
        w.zstd_out_capacities=arena_ptr<std::size_t>(base,zstd.capacities);
        w.zstd_actual_bytes=arena_ptr<std::size_t>(base,zstd.actual);
        w.zstd_out_ptrs=arena_ptr<void*>(base,zstd.out_ptrs);
        w.zstd_statuses=arena_ptr<nvcompStatus_t>(base,zstd.statuses);
        w.zstd_temp=base+zstd.temp;
        w.ans_copy_descs=arena_ptr<DeviceChunkCopyDesc>(base,ans.copies);
        w.ans_in_ptrs=arena_ptr<void*>(base,ans.in_ptrs);
        w.ans_in_bytes=arena_ptr<std::size_t>(base,ans.in_bytes);
        w.ans_out_capacities=arena_ptr<std::size_t>(base,ans.capacities);
        w.ans_actual_bytes=arena_ptr<std::size_t>(base,ans.actual);
        w.ans_out_ptrs=arena_ptr<void*>(base,ans.out_ptrs);
        w.ans_statuses=arena_ptr<nvcompStatus_t>(base,ans.statuses);
        w.ans_temp=base+ans.temp;
        return w;
    }
};
static void check_alignment(nvcompAlignmentRequirements_t a) {
    require(a.input && a.output && a.temp && 256%a.input==0
            && 256%a.output==0 && 256%a.temp==0,"unsupported codec alignment");
}

static py::array_t<float> decode(py::bytes compressed,bool output_soa) {
    const std::string bytes=compressed;
    const std::vector<unsigned char> h_comp(bytes.begin(),bytes.end());
    Header h=parse(h_comp);
    CodecBatchPlan plan=make_plan(h);
    const int N=h.n,D=h.dims,B=h.components[3].n;
    const int ndiff=h.components[1].n,nkeep=h.components[2].n;
    const auto range=std::uint32_t(h.wrap_max)-std::uint32_t(h.wrap_min)+1u;
    const auto dn=mul(N,D);
    const cudaStream_t stream=0;
    require(plan.ans_num_chunks<=INT_MAX && plan.zstd_num_chunks<=INT_MAX,"too many codec chunks");

    // Phase 1: host header/descriptor planning, with all scratch-size queries.
    std::size_t zstd_temp=0,ans_temp=0;
    nvcompAlignmentRequirements_t alignment{};
    if (plan.zstd_num_chunks) {
        const auto opts=nvcompBatchedZstdDecompressDefaultOpts;
        NVCOMP_CHECK(nvcompBatchedZstdDecompressGetRequiredAlignments(opts,&alignment));
        check_alignment(alignment);
        NVCOMP_CHECK(nvcompBatchedZstdDecompressGetTempSizeAsync(
            plan.zstd_num_chunks,plan.zstd_max_chunk,opts,&zstd_temp,plan.zstd_total_uncomp));
    }
    if (plan.ans_num_chunks) {
        auto opts=nvcompBatchedANSDecompressDefaultOpts;
        opts.data_type=NVCOMP_TYPE_UCHAR;
        NVCOMP_CHECK(nvcompBatchedANSDecompressGetRequiredAlignments(opts,&alignment));
        check_alignment(alignment);
        NVCOMP_CHECK(nvcompBatchedANSDecompressGetTempSizeAsync(
            plan.ans_num_chunks,plan.ans_max_chunk,opts,&ans_temp,plan.ans_total_uncomp));
    }
    std::size_t count_temp=0,tag_temp=0,block_temp=0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,count_temp,
        static_cast<int*>(nullptr),static_cast<int*>(nullptr),N,stream));
    const int tag_n=std::max(1,plan.max_tag_n);
    if (plan.max_tag_n) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,tag_temp,
            static_cast<unsigned long long*>(nullptr),static_cast<unsigned long long*>(nullptr),tag_n,stream));
    }
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(nullptr,block_temp,
        static_cast<ResetValue*>(nullptr),static_cast<ResetValue*>(nullptr),ModularResetAdd{range},B,stream));
    const std::size_t scan_capacity=std::max({std::size_t(1),count_temp,tag_temp,block_temp});

    // Phase 2: preserve the original single device-arena allocation. Every
    // intermediate and codec workspace lives until the final D2H completes.
    ArenaLayout layout;
    const auto off_comp=layout.reserve_bytes(h_comp.size(),256,"LCG2 compressed container");
    const CodecOffsets codec_offsets(layout,plan,zstd_temp,ans_temp);
    std::array<std::size_t,5> off_streams;
    for (int s=0;s<5;++s)
        off_streams[s]=layout.reserve<int>(std::max<std::size_t>(1,mul(h.components[s].dims,h.components[s].n)),"LCG2 integer stream");
    const auto off_blocks=layout.reserve<int>(mul(B,D),"LCG2 compact blocks");
    const auto off_flags=layout.reserve<int>(N,"LCG2 flags");
    const auto off_positions=layout.reserve<int>(N,"LCG2 positions");
    const auto off_terms=layout.reserve<ResetValue>(B,"LCG2 block scan inputs");
    const auto off_prefix=layout.reserve<ResetValue>(B,"LCG2 block scan outputs");
    const auto off_bits=layout.reserve<unsigned long long>(tag_n,"LCG2 tag bits");
    const auto off_offsets=layout.reserve<unsigned long long>(tag_n,"LCG2 tag offsets");
    const auto off_scan=layout.reserve_bytes(scan_capacity,256,"LCG2 CUB scratch");
    const auto off_bid=layout.reserve<int>(dn,"LCG2 terminal block IDs");
    const auto off_parent=layout.reserve<int>(N,"LCG2 parents");
    const auto off_parent_tmp=layout.reserve<int>(N,"LCG2 temporary parents");
    const auto off_output=layout.reserve<float>(dn,"LCG2 coordinates");
    const auto off_error=layout.reserve<int>(1,"LCG2 reconstruction status");
    DeviceBuffer<unsigned char> d_arena(layout.bytes);
    auto* arena=d_arena.ptr;
    auto w=codec_offsets.bind(arena);
    auto* d_comp=arena+off_comp;
    auto metadata=build_codec_batch_metadata(plan,d_comp,w);
    std::array<int*,5> values;
    for (int s=0;s<5;++s) values[s]=arena_ptr<int>(arena,off_streams[s]);
    auto* blocks=arena_ptr<int>(arena,off_blocks);
    auto* flags=arena_ptr<int>(arena,off_flags);
    auto* positions=arena_ptr<int>(arena,off_positions);
    auto* terms=arena_ptr<ResetValue>(arena,off_terms);
    auto* prefix=arena_ptr<ResetValue>(arena,off_prefix);
    auto* tag_bits=arena_ptr<unsigned long long>(arena,off_bits);
    auto* tag_offsets=arena_ptr<unsigned long long>(arena,off_offsets);
    void* scratch=arena+off_scan;
    auto* bid=arena_ptr<int>(arena,off_bid);
    auto* parent=arena_ptr<int>(arena,off_parent);
    auto* parent_tmp=arena_ptr<int>(arena,off_parent_tmp);
    auto* output=arena_ptr<float>(arena,off_output);
    auto* error=arena_ptr<int>(arena,off_error);

    ArenaLayout host_layout;
    const auto hz_status=host_layout.reserve<nvcompStatus_t>(plan.zstd_num_chunks,"host Zstd status");
    const auto hz_actual=host_layout.reserve<std::size_t>(plan.zstd_num_chunks,"host Zstd lengths");
    const auto ha_status=host_layout.reserve<nvcompStatus_t>(plan.ans_num_chunks,"host ANS status");
    const auto ha_actual=host_layout.reserve<std::size_t>(plan.ans_num_chunks,"host ANS lengths");
    const auto herror=host_layout.reserve<int>(1,"host reconstruction status");
    const auto houtput=host_layout.reserve<float>(dn,"host coordinate output");
    PinnedHostBuffer<unsigned char> h_arena(host_layout.bytes);
    auto* ha=h_arena.ptr;
    const std::vector<py::ssize_t> shape=output_soa
        ? std::vector<py::ssize_t>{D,N} : std::vector<py::ssize_t>{N,D};
    py::array_t<float> result(shape);

    // Phase 3: exactly one bulk H2D, plus small codec descriptor uploads.
    CUDA_CHECK(cudaMemcpyAsync(d_comp,h_comp.data(),h_comp.size(),cudaMemcpyHostToDevice,stream));
    const auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemsetAsync(error,0,sizeof(int),stream));
    // Failed codec chunks must not leave uninitialized tag bytes to be parsed.
    if (plan.zstd_output_bytes) CUDA_CHECK(cudaMemsetAsync(w.zstd_outputs,0,plan.zstd_output_bytes,stream));
    if (plan.ans_output_bytes) CUDA_CHECK(cudaMemsetAsync(w.ans_outputs,0,plan.ans_output_bytes,stream));
    upload_codec_batch_metadata(metadata,w,stream);
    // LCG2 ANS sections are independent of Zstd sections. The existing order
    // safely reuses the aligned-input arena; no ANS/Zstd nesting is introduced.
    enqueue_global_codec_batches(plan,w,zstd_temp,ans_temp,stream,true);

    // Phase 4: integer decode, including byte-rANS Raw and byte-sized tags.
    for (int s=0;s<5;++s)
        unpack_component(h.components[s],w,values[s],tag_bits,tag_offsets,scratch,scan_capacity,error,stream);
    const auto t1 = std::chrono::steady_clock::now();

    // Phase 5: choice2 ranks -> inverse ZigZag/wrap -> parallel reset scan.
    flags_kernel<<<grid_for(B),256,0,stream>>>(values[3],B,true,flags,error);
    CUDA_CHECK(cudaGetLastError());
    {
        auto available=scan_capacity;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scratch,available,flags,positions,B,stream));
    }
    check_count_kernel<<<1,1,0,stream>>>(flags,positions,B,ndiff,error);
    CUDA_CHECK(cudaGetLastError());
    for (int d=0;d<D;++d) {
        block_terms_kernel<<<grid_for(B),256,0,stream>>>(values[3],positions,
            values[1]+std::size_t(d)*ndiff,values[2]+std::size_t(d)*nkeep,
            B,ndiff,nkeep,h.wrap_min,range,terms,error);
        CUDA_CHECK(cudaGetLastError());
        auto available=scan_capacity;
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(scratch,available,terms,prefix,ModularResetAdd{range},B,stream));
        store_blocks_kernel<<<grid_for(B),256,0,stream>>>(prefix,B,h.wrap_min,blocks+std::size_t(d)*B);
        CUDA_CHECK(cudaGetLastError());
    }
    const auto t2 = std::chrono::steady_clock::now();

    // Phase 6: full-length IDRel uses zero for an explicit block. Retain the
    // original pointer-jumping algorithm, with no separate stored choice stream.
    flags_kernel<<<grid_for(N),256,0,stream>>>(values[0],N,false,flags,error);
    CUDA_CHECK(cudaGetLastError());
    {
        auto available=scan_capacity;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scratch,available,flags,positions,N,stream));
    }
    check_count_kernel<<<1,1,0,stream>>>(flags,positions,N,B,error);
    CUDA_CHECK(cudaGetLastError());
    init_chain_kernel<<<grid_for(N),256,0,stream>>>(values[0],positions,blocks,N,D,B,bid,parent,error);
    CUDA_CHECK(cudaGetLastError());
    const unsigned jump_grid=static_cast<unsigned>((std::size_t(N)+255)/256);
    for (unsigned long long span=1;span<static_cast<unsigned>(N);span*=2) {
        pointer_jump_kernel<<<jump_grid,256,0,stream>>>(parent,parent_tmp,N);
        CUDA_CHECK(cudaGetLastError());
        std::swap(parent,parent_tmp);
    }
    const auto t3 = std::chrono::steady_clock::now();
    // Read immutable terminal block IDs directly, avoiding an in-place bid
    // gather. Exact integer reconstruction precedes the specified float FMA.
    reconstruct_kernel<<<grid_for(dn),256,0,stream>>>(bid,parent,values[4],d_comp,h.minima_off,
        N,D,h.p,h.eb,output_soa,output,error);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto t4 = std::chrono::steady_clock::now();

    // Phase 7: all status copies and the ONLY bulk D2H are grouped at the end.
    if (plan.zstd_num_chunks) {
        CUDA_CHECK(cudaMemcpyAsync(ha+hz_status,w.zstd_statuses,mul(plan.zstd_num_chunks,sizeof(nvcompStatus_t)),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaMemcpyAsync(ha+hz_actual,w.zstd_actual_bytes,mul(plan.zstd_num_chunks,sizeof(std::size_t)),cudaMemcpyDeviceToHost,stream));
    }
    if (plan.ans_num_chunks) {
        CUDA_CHECK(cudaMemcpyAsync(ha+ha_status,w.ans_statuses,mul(plan.ans_num_chunks,sizeof(nvcompStatus_t)),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaMemcpyAsync(ha+ha_actual,w.ans_actual_bytes,mul(plan.ans_num_chunks,sizeof(std::size_t)),cudaMemcpyDeviceToHost,stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(ha+herror,error,sizeof(int),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaMemcpyAsync(ha+houtput,output,mul(dn,sizeof(float)),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    validate_batch_results_host(arena_ptr<nvcompStatus_t>(ha,hz_status),arena_ptr<std::size_t>(ha,hz_actual),metadata.zstd_out_capacities,"Zstd");
    validate_batch_results_host(arena_ptr<nvcompStatus_t>(ha,ha_status),arena_ptr<std::size_t>(ha,ha_actual),metadata.ans_out_capacities,"ANS");
    const int reconstruction_status=*arena_ptr<int>(ha,herror);
    if (reconstruction_status) throw std::runtime_error(
        "LCG2: invalid integer payload or reconstruction, error flags="+std::to_string(reconstruction_status));
    std::memcpy(result.mutable_data(),ha+houtput,mul(dn,sizeof(float)));
    const float decode = std::chrono::duration<float>(t1-t0).count();
    const float l2 = std::chrono::duration<float>(t2-t1).count();
    const float l1 = std::chrono::duration<float>(t3-t2).count();
    const float qb = std::chrono::duration<float>(t4-t3).count();
    std::printf("\t%.6f\n\t%.6f\n\t%.6f\n\t%.6f\n", qb, l1, l2, decode);
    return result;
}
} // namespace lcg2

static py::array lcps_decompress_gpu_dispatch(py::bytes compressed,bool output_soa=false) {
    char* data=nullptr; Py_ssize_t size=0;
    if (PyBytes_AsStringAndSize(compressed.ptr(),&data,&size)!=0) throw py::error_already_set();
    if (size>=4 && std::memcmp(data,"LCG2",4)==0) return lcg2::decode(compressed,output_soa);
    if (size>=4 && std::memcmp(data,"LCG1",4)==0) return lcps_decompress_gpu_ans(compressed,output_soa);
    throw std::runtime_error("GPU decoder expects LCG2 or legacy LCG1; CPU LCP2 is a different format");
}

PYBIND11_MODULE(lcps_gpu_decompress_ans, m) {
    m.doc() = "CUDA decompressor for current LCG2 and legacy LCG1 streams";
    m.def(
        "lcps_decompress_gpu",
        &lcps_decompress_gpu_dispatch,
        py::arg("compressed_data"),
        py::arg("output_soa") = false,
        R"pbdoc(
Decompress a current LCG2 or legacy LCG1 GPU bitstream in original point order.
LCG2 supports Tagged+Zstd (ANS tags only), Fixed+Zstd, Raw+ANS/UCHAR, and empty
streams. It decodes IDRel/diff/keep/choice2/rel and returns float32 coordinates.
LCG1 retains its existing encodings and float64 output. output_soa=False returns
(N,D); True returns (D,N), both C-contiguous.
Host parses headers serially, then allocates one device arena. All Zstd chunks
share one batch, followed by one global ANS batch. LCG2 has no ANS/Zstd nesting.
The compressed container has one bulk H2D; reconstructed coordinates have one
bulk D2H; status metadata uses only small transfers; and the stream synchronizes
once after the final D2H.
LCG2 v1 reproduces the compressor's float32 midpoint/FMA reconstruction. Its
float32 quantization can lose more than eb, and it stores no exact-coordinate
corrections; a decoder alone cannot guarantee a strict original-coordinate eb.
)pbdoc"
    );
}
