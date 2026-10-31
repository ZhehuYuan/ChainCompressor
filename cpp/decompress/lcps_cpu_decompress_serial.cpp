#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <zstd.h>

#if defined(__FAST_MATH__)
#error "LCP2 decoding requires compilation without -ffast-math"
#endif

static bool bytes_starts_with(const std::vector<unsigned char>& data, const char* magic, std::size_t len) {
    if (data.size() < len) return false;
    return std::memcmp(data.data(), magic, len) == 0;
}

static bool looks_like_raw_component_payload(const std::vector<unsigned char>& data) {
    return bytes_starts_with(data, "FIXD", 4) ||
           bytes_starts_with(data, "TAG3", 4) ||
           bytes_starts_with(data, "RANS1", 5) ||
           bytes_starts_with(data, "BANS", 4) ||
           bytes_starts_with(data, "UANS", 4) ||
           bytes_starts_with(data, "ZFXD", 4) ||
           bytes_starts_with(data, "ZFMX", 4) ||
           bytes_starts_with(data, "ZTG3", 4) ||
           bytes_starts_with(data, "ZRNS", 4) ||
           bytes_starts_with(data, "ZRMX", 4) ||
           bytes_starts_with(data, "EMPTY", 5);
}

static std::vector<unsigned char> zstd_decompress_cpp(const std::vector<unsigned char>& compressed) {
    unsigned long long frame_size = ZSTD_getFrameContentSize(
        compressed.data(),
        compressed.size()
    );

    if (frame_size == ZSTD_CONTENTSIZE_ERROR) {
        throw std::runtime_error("component payload is neither raw LCPS payload nor valid Zstd frame");
    }
    if (frame_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        throw std::runtime_error(
            "Zstd frame content size is unknown. "
            "The compressor should use ZSTD_compress(), which stores the content size."
        );
    }
    if (frame_size > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Zstd decompressed component is too large for this platform");
    }

    const std::size_t expected_size = static_cast<std::size_t>(frame_size);
    // Keep a non-null destination pointer even for a valid empty Zstd frame.
    std::vector<unsigned char> out(expected_size == 0 ? 1 : expected_size);
    const std::size_t decompressed_size = ZSTD_decompress(
        out.data(),
        expected_size,
        compressed.data(),
        compressed.size()
    );

    if (ZSTD_isError(decompressed_size)) {
        throw std::runtime_error(
            std::string("ZSTD_decompress failed: ") + ZSTD_getErrorName(decompressed_size)
        );
    }
    if (decompressed_size != expected_size) {
        throw std::runtime_error("Zstd decompressed size mismatch");
    }
    out.resize(expected_size);
    return out;
}

static std::vector<unsigned char> maybe_zstd_decompress_component_payload(const std::vector<unsigned char>& payload) {
    // Backward compatibility: some older compressors wrapped an entire
    // component in one Zstd frame. Current formats keep their custom headers
    // visible and compress only the internal data body.
    if (looks_like_raw_component_payload(payload)) {
        return payload;
    }
    return zstd_decompress_cpp(payload);
}



namespace py = pybind11;

class Reader {
public:
    const unsigned char* p = nullptr;
    std::size_t n = 0;
    std::size_t pos = 0;

    Reader() = default;
    Reader(const unsigned char* data, std::size_t size) : p(data), n(size) {}

    std::size_t remaining() const { return n - pos; }

    void require(std::size_t k) const {
        if (k > n - pos) throw std::runtime_error("truncated compressed stream");
    }

    bool starts_with(const char* magic, std::size_t k) const {
        if (remaining() < k) return false;
        return std::memcmp(p + pos, magic, k) == 0;
    }

    void expect_magic(const char* magic, std::size_t k) {
        require(k);
        if (std::memcmp(p + pos, magic, k) != 0) {
            throw std::runtime_error(std::string("expected magic ") + std::string(magic, k));
        }
        pos += k;
    }

    std::string read_magic4() {
        require(4);
        std::string s(reinterpret_cast<const char*>(p + pos), 4);
        pos += 4;
        return s;
    }

    template <typename T>
    T read_le() {
        static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "scalar only");
        require(sizeof(T));
        unsigned char buf[sizeof(T)];
        std::memcpy(buf, p + pos, sizeof(T));
        pos += sizeof(T);
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        std::reverse(buf, buf + sizeof(T));
#endif
        T v;
        std::memcpy(&v, buf, sizeof(T));
        return v;
    }

    std::vector<unsigned char> read_bytes(std::size_t k) {
        require(k);
        std::vector<unsigned char> out(p + pos, p + pos + k);
        pos += k;
        return out;
    }

    Reader subreader(std::size_t k) {
        require(k);
        Reader r(p + pos, k);
        pos += k;
        return r;
    }
};

static std::size_t u64_to_size_checked(std::uint64_t v, const char* field) {
    if (v > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string(field) + " is too large");
    }
    return static_cast<std::size_t>(v);
}

static std::size_t checked_add_size(std::size_t a, std::size_t b, const char* field) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::runtime_error(std::string(field) + " size overflow");
    }
    return a + b;
}

static std::size_t checked_mul_size(std::size_t a, std::size_t b, const char* field) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(std::string(field) + " size overflow");
    }
    return a * b;
}

static std::size_t fixed_payload_bytes_checked(
    std::size_t n,
    int width,
    const char* field
) {
    if (width <= 0 || width > 64) {
        throw std::runtime_error(std::string("invalid ") + field + " width");
    }
    const std::size_t bits = checked_mul_size(
        n, static_cast<std::size_t>(width), field
    );
    return checked_add_size(bits, 7, field) / 8;
}

static std::vector<unsigned char> read_zstd_body_exact(
    Reader& r,
    std::size_t raw_bytes,
    std::size_t compressed_bytes,
    const char* field
) {
    std::vector<unsigned char> compressed = r.read_bytes(compressed_bytes);
    std::vector<unsigned char> raw = zstd_decompress_cpp(compressed);
    if (raw.size() != raw_bytes) {
        throw std::runtime_error(std::string(field) + " raw byte count mismatch");
    }
    return raw;
}

static inline int bit_length_u64(std::uint64_t v) {
    if (v == 0) return 0;
#if defined(__GNUG__) || defined(__clang__)
    return 64 - __builtin_clzll(v);
#else
    int bits = 0;
    while (v != 0) { ++bits; v >>= 1; }
    return bits;
#endif
}

static std::uint64_t read_bits_msb_first(const unsigned char* payload, std::size_t payload_size, std::uint64_t bit_offset, int width) {
    if (width < 0 || width > 64) throw std::runtime_error("unsupported bit width");
    std::uint64_t v = 0;
    for (int k = 0; k < width; ++k) {
        const std::uint64_t bit_pos = bit_offset + static_cast<std::uint64_t>(k);
        const std::uint64_t byte_idx = bit_pos >> 3;
        const int bit_in_byte = 7 - static_cast<int>(bit_pos & 7ULL);
        if (byte_idx >= payload_size) throw std::runtime_error("bit payload truncated");
        const std::uint64_t bit = (payload[byte_idx] >> bit_in_byte) & 1u;
        v = (v << 1) | bit;
    }
    return v;
}

static std::vector<std::int64_t> decode_fixed_values(
    std::int64_t mn,
    std::size_t n,
    int width,
    const unsigned char* payload,
    std::size_t payload_bytes
) {
    const std::size_t expected = fixed_payload_bytes_checked(n, width, "fixed payload");
    if (payload_bytes != expected) {
        throw std::runtime_error("fixed payload byte count mismatch");
    }
    std::vector<std::int64_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t bit_offset =
            static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(width);
        const std::uint64_t delta = read_bits_msb_first(
            payload, payload_bytes, bit_offset, width
        );
        out[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(mn) + delta);
    }
    return out;
}

static std::vector<std::int64_t> decode_fixed_stream(Reader& r) {
    r.expect_magic("FIXD", 4);
    const std::int64_t mn = r.read_le<std::int64_t>();
    const std::uint64_t n64 = r.read_le<std::uint64_t>();
    const std::uint8_t width_u8 = r.read_le<std::uint8_t>();
    const int width = static_cast<int>(width_u8);
    if (width <= 0 || width > 64) throw std::runtime_error("invalid FIXD width");
    const std::size_t n = u64_to_size_checked(n64, "FIXD n");
    const std::size_t payload_bytes = fixed_payload_bytes_checked(n, width, "FIXD payload");
    std::vector<unsigned char> payload = r.read_bytes(payload_bytes);
    return decode_fixed_values(mn, n, width, payload.data(), payload.size());
}

static std::vector<std::int64_t> decode_zfixed_stream(Reader& r) {
    r.expect_magic("ZFXD", 4);
    const std::int64_t mn = r.read_le<std::int64_t>();
    const std::size_t n = u64_to_size_checked(r.read_le<std::uint64_t>(), "ZFXD n");
    const int width = static_cast<int>(r.read_le<std::uint8_t>());
    const std::size_t raw_bytes = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZFXD raw payload bytes"
    );
    const std::size_t compressed_bytes = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZFXD compressed payload bytes"
    );
    const std::size_t expected_raw = fixed_payload_bytes_checked(n, width, "ZFXD payload");
    if (raw_bytes != expected_raw) {
        throw std::runtime_error("ZFXD raw payload byte count mismatch");
    }
    std::vector<unsigned char> payload = read_zstd_body_exact(
        r, raw_bytes, compressed_bytes, "ZFXD payload"
    );
    return decode_fixed_values(mn, n, width, payload.data(), payload.size());
}

struct RansSymbolInfo {
    std::int64_t sym = 0;
    std::uint32_t freq = 0;
    std::uint32_t start = 0;
};

static std::vector<std::int64_t> decode_rans1_stream(Reader& r) {
    r.expect_magic("RANS1", 5);
    const int precision = static_cast<int>(r.read_le<std::uint8_t>());
    if (precision <= 0 || precision > 20) throw std::runtime_error("unsupported RANS precision");
    const std::uint64_t n64 = r.read_le<std::uint64_t>();
    const std::uint32_t num_syms = r.read_le<std::uint32_t>();
    constexpr std::uint32_t RANS_L_CPP = 1u << 23;
    const std::uint64_t total = 1ull << precision;

    std::vector<std::pair<std::int64_t, std::uint32_t>> freqs;
    freqs.reserve(num_syms);
    for (std::uint32_t k = 0; k < num_syms; ++k) {
        std::int64_t sym = r.read_le<std::int64_t>();
        std::uint32_t freq = r.read_le<std::uint32_t>();
        if (freq == 0) throw std::runtime_error("RANS zero frequency");
        freqs.emplace_back(sym, freq);
    }
    std::uint32_t state = r.read_le<std::uint32_t>();

    if (n64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("RANS stream too large");
    }
    const std::size_t n = static_cast<std::size_t>(n64);
    if (n == 0) {
        return {};
    }
    if (num_syms == 0) throw std::runtime_error("non-empty RANS stream has no symbols");

    std::vector<RansSymbolInfo> table(total);
    std::uint64_t cum = 0;
    for (const auto& kv : freqs) {
        const std::int64_t sym = kv.first;
        const std::uint32_t freq = kv.second;
        if (cum + freq > total) throw std::runtime_error("RANS cumulative frequency overflow");
        for (std::uint32_t x = 0; x < freq; ++x) {
            table[static_cast<std::size_t>(cum + x)] = RansSymbolInfo{sym, freq, static_cast<std::uint32_t>(cum)};
        }
        cum += freq;
    }
    if (cum != total) throw std::runtime_error("RANS normalized total mismatch");

    std::vector<std::int64_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t x = state & static_cast<std::uint32_t>(total - 1);
        const RansSymbolInfo& info = table[x];
        out[i] = info.sym;
        state = info.freq * (state >> precision) + (x - info.start);
        while (state < RANS_L_CPP && r.remaining() > 0) {
            state = static_cast<std::uint32_t>((static_cast<std::uint64_t>(state) << 8) | r.read_le<std::uint8_t>());
        }
    }
    return out;
}

static std::vector<std::int64_t> decode_zrans_stream(Reader& r) {
    r.expect_magic("ZRNS", 4);
    const int precision = static_cast<int>(r.read_le<std::uint8_t>());
    if (precision <= 0 || precision >= 31) {
        throw std::runtime_error("unsupported ZRNS precision");
    }
    const std::size_t n = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZRNS n"
    );
    const std::uint32_t num_syms = r.read_le<std::uint32_t>();
    std::uint32_t state = r.read_le<std::uint32_t>();
    const std::size_t raw_body_bytes = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZRNS raw body bytes"
    );
    const std::size_t compressed_body_bytes = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZRNS compressed body bytes"
    );

    const std::size_t codebook_bytes = checked_mul_size(
        static_cast<std::size_t>(num_syms), std::size_t(12), "ZRNS codebook"
    );
    if (codebook_bytes > raw_body_bytes) {
        throw std::runtime_error("ZRNS body is shorter than its codebook");
    }
    std::vector<unsigned char> body = read_zstd_body_exact(
        r, raw_body_bytes, compressed_body_bytes, "ZRNS body"
    );
    Reader br(body.data(), body.size());

    struct Entry {
        std::int64_t sym = 0;
        std::uint32_t freq = 0;
        std::uint32_t start = 0;
        std::uint32_t end = 0;
    };
    std::vector<Entry> entries;
    std::vector<std::uint32_t> cumulative_ends;
    entries.reserve(num_syms);
    cumulative_ends.reserve(num_syms);

    const std::uint64_t total = std::uint64_t(1) << precision;
    std::uint64_t cumulative = 0;
    std::int64_t previous_symbol = 0;
    bool have_previous_symbol = false;
    for (std::uint32_t k = 0; k < num_syms; ++k) {
        const std::int64_t sym = br.read_le<std::int64_t>();
        const std::uint32_t freq = br.read_le<std::uint32_t>();
        if (freq == 0) throw std::runtime_error("ZRNS zero frequency");
        if (have_previous_symbol && sym <= previous_symbol) {
            throw std::runtime_error("ZRNS codebook symbols are not strictly increasing");
        }
        if (cumulative + freq > total) {
            throw std::runtime_error("ZRNS cumulative frequency overflow");
        }
        Entry e;
        e.sym = sym;
        e.freq = freq;
        e.start = static_cast<std::uint32_t>(cumulative);
        cumulative += freq;
        e.end = static_cast<std::uint32_t>(cumulative);
        entries.push_back(e);
        cumulative_ends.push_back(e.end);
        previous_symbol = sym;
        have_previous_symbol = true;
    }

    constexpr std::uint32_t RANS_L_CPP = 1u << 23;
    if (n == 0) {
        if (num_syms != 0 || cumulative != 0 || br.remaining() != 0
            || state != RANS_L_CPP) {
            throw std::runtime_error("invalid empty ZRNS stream");
        }
        return {};
    }
    if (num_syms == 0 || cumulative != total) {
        throw std::runtime_error("ZRNS normalized total mismatch");
    }

    std::vector<std::int64_t> out(n);
    const std::uint32_t mask = static_cast<std::uint32_t>(total - 1);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t x = state & mask;
        auto it = std::upper_bound(cumulative_ends.begin(), cumulative_ends.end(), x);
        if (it == cumulative_ends.end()) {
            throw std::runtime_error("ZRNS state does not map to a symbol");
        }
        const Entry& e = entries[static_cast<std::size_t>(it - cumulative_ends.begin())];
        out[i] = e.sym;
        const std::uint64_t next_state =
            static_cast<std::uint64_t>(e.freq) * (state >> precision)
            + (x - e.start);
        if (next_state > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("ZRNS state overflow");
        }
        state = static_cast<std::uint32_t>(next_state);
        while (state < RANS_L_CPP) {
            if (br.remaining() == 0) {
                throw std::runtime_error("ZRNS renormalization bytes are truncated");
            }
            state = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(state) << 8)
                | br.read_le<std::uint8_t>()
            );
        }
    }
    if (br.remaining() != 0) throw std::runtime_error("ZRNS body has trailing bytes");
    if (state != RANS_L_CPP) throw std::runtime_error("ZRNS final decoder state mismatch");
    return out;
}

static std::vector<std::int64_t> decode_bans_stream_cpu_unavailable(Reader& r) {
    (void)r;
    throw std::runtime_error(
        "BANS/UANS is an nvCOMP ANS blob. A pure serial CPU decoder cannot decode it. "
        "Use a nvCOMP/GPU decoder, or make the compressor emit CPU RANS1 streams instead."
    );
}

static std::vector<std::int64_t> decode_uans_stream_cpu_unavailable(Reader& r) {
    (void)r;
    throw std::runtime_error(
        "UANS is an nvCOMP ANS blob. A pure serial CPU decoder cannot decode it. "
        "Use a nvCOMP/GPU decoder, or make the compressor emit CPU RANS1 streams instead."
    );
}

static std::vector<std::int64_t> decode_one_stream(Reader& r) {
    if (r.starts_with("ZFXD", 4)) return decode_zfixed_stream(r);
    if (r.starts_with("ZRNS", 4)) return decode_zrans_stream(r);
    if (r.starts_with("FIXD", 4)) return decode_fixed_stream(r);
    if (r.starts_with("RANS1", 5)) return decode_rans1_stream(r);
    if (r.starts_with("BANS", 4)) return decode_bans_stream_cpu_unavailable(r);
    if (r.starts_with("UANS", 4)) return decode_uans_stream_cpu_unavailable(r);
    throw std::runtime_error("unknown stream encoding magic");
}

static std::vector<std::int64_t> decode_zfixed_matrix(
    Reader& r,
    int expected_D,
    std::size_t expected_n
) {
    r.expect_magic("ZFMX", 4);
    const std::uint32_t D_u32 = r.read_le<std::uint32_t>();
    const std::size_t n = u64_to_size_checked(r.read_le<std::uint64_t>(), "ZFMX n");
    if (D_u32 != static_cast<std::uint32_t>(expected_D) || n != expected_n) {
        throw std::runtime_error("ZFMX D/n metadata mismatch");
    }

    struct Field {
        std::int64_t mn = 0;
        int width = 0;
        std::size_t offset = 0;
        std::size_t bytes = 0;
    };
    std::vector<Field> fields(static_cast<std::size_t>(expected_D));
    std::size_t raw_sum = 0;
    for (int d = 0; d < expected_D; ++d) {
        Field& f = fields[static_cast<std::size_t>(d)];
        f.mn = r.read_le<std::int64_t>();
        f.width = static_cast<int>(r.read_le<std::uint8_t>());
        f.offset = raw_sum;
        f.bytes = u64_to_size_checked(
            r.read_le<std::uint64_t>(), "ZFMX dimension payload bytes"
        );
        const std::size_t expected_bytes = fixed_payload_bytes_checked(
            n, f.width, "ZFMX dimension payload"
        );
        if (f.bytes != expected_bytes) {
            throw std::runtime_error("ZFMX dimension payload byte count mismatch");
        }
        raw_sum = checked_add_size(raw_sum, f.bytes, "ZFMX raw payload");
    }
    const std::size_t raw_bytes = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZFMX total raw payload bytes"
    );
    if (raw_bytes != raw_sum) throw std::runtime_error("ZFMX raw payload total mismatch");
    const std::size_t compressed_bytes = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZFMX compressed payload bytes"
    );
    std::vector<unsigned char> payload = read_zstd_body_exact(
        r, raw_bytes, compressed_bytes, "ZFMX payload"
    );

    const std::size_t flat_size = checked_mul_size(
        static_cast<std::size_t>(expected_D), n, "ZFMX output"
    );
    std::vector<std::int64_t> flat(flat_size);
    for (int d = 0; d < expected_D; ++d) {
        const Field& f = fields[static_cast<std::size_t>(d)];
        const unsigned char* field_payload =
            f.bytes == 0 ? nullptr : payload.data() + f.offset;
        std::vector<std::int64_t> one = decode_fixed_values(
            f.mn, n, f.width, field_payload, f.bytes
        );
        std::copy(
            one.begin(), one.end(),
            flat.begin() + static_cast<std::size_t>(d) * n
        );
    }
    return flat;
}

static std::vector<std::int64_t> decode_zrans_matrix(
    Reader& r,
    int expected_D,
    std::size_t expected_n
) {
    r.expect_magic("ZRMX", 4);
    const std::uint32_t D_u32 = r.read_le<std::uint32_t>();
    const std::size_t n = u64_to_size_checked(r.read_le<std::uint64_t>(), "ZRMX n");
    const std::size_t zrns_bytes = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZRMX ZRNS bytes"
    );
    if (D_u32 != static_cast<std::uint32_t>(expected_D) || n != expected_n) {
        throw std::runtime_error("ZRMX D/n metadata mismatch");
    }
    Reader zr = r.subreader(zrns_bytes);
    std::vector<std::int64_t> point_major = decode_zrans_stream(zr);
    if (zr.remaining() != 0) throw std::runtime_error("ZRMX ZRNS payload has trailing bytes");
    const std::size_t expected_values = checked_mul_size(
        static_cast<std::size_t>(expected_D), n, "ZRMX output"
    );
    if (point_major.size() != expected_values) {
        throw std::runtime_error("ZRMX point-major value count mismatch");
    }
    std::vector<std::int64_t> flat(expected_values);
    for (std::size_t i = 0; i < n; ++i) {
        for (int d = 0; d < expected_D; ++d) {
            flat[static_cast<std::size_t>(d) * n + i] =
                point_major[i * static_cast<std::size_t>(expected_D) + static_cast<std::size_t>(d)];
        }
    }
    return flat;
}

static std::vector<std::int64_t> decode_ztagged_matrix(Reader& r, int D) {
    r.expect_magic("ZTG3", 4);
    if (D <= 0) throw std::runtime_error("invalid D for ZTG3");
    const std::size_t n = u64_to_size_checked(r.read_le<std::uint64_t>(), "ZTG3 n");
    const int max_width = static_cast<int>(r.read_le<std::uint8_t>());
    if ((n == 0 && max_width != 0) || (n > 0 && (max_width <= 0 || max_width > 64))) {
        throw std::runtime_error("invalid ZTG3 max width");
    }
    const std::size_t tag_payload_bytes = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZTG3 tag payload bytes"
    );
    Reader tr = r.subreader(tag_payload_bytes);
    std::vector<std::int64_t> tags = decode_one_stream(tr);
    if (tr.remaining() != 0) throw std::runtime_error("ZTG3 tag payload has trailing bytes");
    if (tags.size() != n) throw std::runtime_error("ZTG3 tag stream length mismatch");

    const std::size_t value_raw_bytes = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZTG3 value raw bytes"
    );
    const std::size_t value_compressed_bytes = u64_to_size_checked(
        r.read_le<std::uint64_t>(), "ZTG3 value compressed bytes"
    );
    std::vector<unsigned char> payload = read_zstd_body_exact(
        r, value_raw_bytes, value_compressed_bytes, "ZTG3 value payload"
    );

    std::size_t total_bits = 0;
    int observed_max_width = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (tags[i] <= 0 || tags[i] > 64) {
            throw std::runtime_error("invalid ZTG3 tag width");
        }
        const int width = static_cast<int>(tags[i]);
        observed_max_width = std::max(observed_max_width, width);
        const std::size_t point_bits = checked_mul_size(
            static_cast<std::size_t>(width), static_cast<std::size_t>(D),
            "ZTG3 point payload"
        );
        total_bits = checked_add_size(total_bits, point_bits, "ZTG3 value payload");
    }
    if (observed_max_width != max_width) {
        throw std::runtime_error("ZTG3 recorded max width does not match tag stream");
    }
    const std::size_t expected_value_bytes = checked_add_size(
        total_bits, 7, "ZTG3 value payload"
    ) / 8;
    if (value_raw_bytes != expected_value_bytes) {
        throw std::runtime_error("ZTG3 value payload byte count mismatch");
    }

    const std::size_t flat_size = checked_mul_size(
        static_cast<std::size_t>(D), n, "ZTG3 output"
    );
    std::vector<std::int64_t> flat(flat_size, 0);
    std::uint64_t bitpos = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int width = static_cast<int>(tags[i]);
        for (int d = 0; d < D; ++d) {
            const std::uint64_t v = read_bits_msb_first(
                payload.data(), payload.size(), bitpos, width
            );
            flat[static_cast<std::size_t>(d) * n + i] = static_cast<std::int64_t>(v);
            bitpos += static_cast<std::uint64_t>(width);
        }
    }
    return flat;
}

static std::vector<std::int64_t> decode_tagged_matrix(Reader& r, int D) {
    r.expect_magic("TAG3", 4);
    if (D <= 0) throw std::runtime_error("invalid D for TAG3");
    const std::uint64_t n64 = r.read_le<std::uint64_t>();
    const int max_width = static_cast<int>(r.read_le<std::uint8_t>());
    (void)max_width;
    const std::uint64_t tag_payload_len64 = r.read_le<std::uint64_t>();
    if (n64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        tag_payload_len64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("TAG3 field too large");
    }
    const std::size_t n = static_cast<std::size_t>(n64);
    const std::size_t tag_payload_len = static_cast<std::size_t>(tag_payload_len64);

    std::vector<std::int64_t> tags;
    if (tag_payload_len > 0) {
        Reader tr = r.subreader(tag_payload_len);
        tags = decode_one_stream(tr);
        if (tr.remaining() != 0) throw std::runtime_error("TAG3 tag payload has trailing bytes");
    }
    if (tags.size() != n) throw std::runtime_error("TAG3 tag stream length mismatch");

    std::uint64_t total_bits = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (tags[i] <= 0 || tags[i] > 64) throw std::runtime_error("invalid TAG3 tag width");
        total_bits += static_cast<std::uint64_t>(tags[i]) * static_cast<std::uint64_t>(D);
    }
    const std::size_t value_bytes = static_cast<std::size_t>((total_bits + 7) / 8);
    std::vector<unsigned char> payload = r.read_bytes(value_bytes);

    std::vector<std::int64_t> flat(static_cast<std::size_t>(D) * n, 0);
    std::uint64_t bitpos = 0;
    for (std::size_t i = 0; i < n; ++i) {
        int w = static_cast<int>(tags[i]);
        for (int d = 0; d < D; ++d) {
            std::uint64_t v = read_bits_msb_first(payload.data(), payload.size(), bitpos, w);
            flat[static_cast<std::size_t>(d) * n + i] = static_cast<std::int64_t>(v);
            bitpos += static_cast<std::uint64_t>(w);
        }
    }
    return flat;
}

static std::vector<std::int64_t> decode_matrix_component_payload(Reader& r, int D, std::size_t expected_n) {
    if (D <= 0) throw std::runtime_error("invalid D");
    if (r.starts_with("EMPTY", 5)) {
        r.expect_magic("EMPTY", 5);
        if (expected_n != 0) throw std::runtime_error("EMPTY matrix has nonzero expected length");
        return {};
    }
    if (r.starts_with("ZFMX", 4)) {
        return decode_zfixed_matrix(r, D, expected_n);
    }
    if (r.starts_with("ZTG3", 4)) {
        std::vector<std::int64_t> flat = decode_ztagged_matrix(r, D);
        if (flat.size() != checked_mul_size(
                static_cast<std::size_t>(D), expected_n, "ZTG3 matrix")) {
            throw std::runtime_error("ZTG3 matrix size mismatch");
        }
        return flat;
    }
    if (r.starts_with("ZRMX", 4)) {
        return decode_zrans_matrix(r, D, expected_n);
    }
    if (D == 1 && r.starts_with("ZFXD", 4)) {
        std::vector<std::int64_t> one = decode_zfixed_stream(r);
        if (one.size() != expected_n) throw std::runtime_error("ZFXD matrix stream length mismatch");
        return one;
    }
    if (D == 1 && r.starts_with("ZRNS", 4)) {
        std::vector<std::int64_t> one = decode_zrans_stream(r);
        if (one.size() != expected_n) throw std::runtime_error("ZRNS matrix stream length mismatch");
        return one;
    }
    if (r.starts_with("TAG3", 4)) {
        std::vector<std::int64_t> flat = decode_tagged_matrix(r, D);
        if (flat.size() != static_cast<std::size_t>(D) * expected_n) {
            throw std::runtime_error("TAG3 matrix size mismatch");
        }
        return flat;
    }
    if (r.starts_with("FIXD", 4)) {
        std::vector<std::int64_t> flat(static_cast<std::size_t>(D) * expected_n);
        for (int d = 0; d < D; ++d) {
            std::vector<std::int64_t> one = decode_fixed_stream(r);
            if (one.size() != expected_n) throw std::runtime_error("FIXD matrix stream length mismatch");
            std::copy(one.begin(), one.end(), flat.begin() + static_cast<std::size_t>(d) * expected_n);
        }
        return flat;
    }
    if (r.starts_with("RANS1", 5)) {
        std::vector<std::int64_t> flat(static_cast<std::size_t>(D) * expected_n);
        for (int d = 0; d < D; ++d) {
            std::vector<std::int64_t> one = decode_rans1_stream(r);
            if (one.size() != expected_n) throw std::runtime_error("RANS matrix stream length mismatch");
            std::copy(one.begin(), one.end(), flat.begin() + static_cast<std::size_t>(d) * expected_n);
        }
        return flat;
    }
    if (r.starts_with("BANS", 4) || r.starts_with("UANS", 4)) {
        return decode_bans_stream_cpu_unavailable(r);
    }
    throw std::runtime_error("unknown matrix component encoding");
}

static std::vector<std::int64_t> decode_scalar_component_payload(Reader& r, std::size_t expected_n) {
    if (r.starts_with("EMPTY", 5)) {
        r.expect_magic("EMPTY", 5);
        if (expected_n != 0) throw std::runtime_error("EMPTY scalar has nonzero expected length");
        return {};
    }
    if (r.starts_with("ZFMX", 4)) {
        return decode_zfixed_matrix(r, 1, expected_n);
    }
    if (r.starts_with("ZTG3", 4)) {
        std::vector<std::int64_t> flat = decode_ztagged_matrix(r, 1);
        if (flat.size() != expected_n) throw std::runtime_error("ZTG3 scalar size mismatch");
        return flat;
    }
    if (r.starts_with("ZRMX", 4)) {
        return decode_zrans_matrix(r, 1, expected_n);
    }
    if (r.starts_with("ZFXD", 4)) {
        std::vector<std::int64_t> one = decode_zfixed_stream(r);
        if (one.size() != expected_n) throw std::runtime_error("ZFXD scalar length mismatch");
        return one;
    }
    if (r.starts_with("ZRNS", 4)) {
        std::vector<std::int64_t> one = decode_zrans_stream(r);
        if (one.size() != expected_n) throw std::runtime_error("ZRNS scalar length mismatch");
        return one;
    }
    if (r.starts_with("TAG3", 4)) {
        std::vector<std::int64_t> flat = decode_tagged_matrix(r, 1);
        if (flat.size() != expected_n) throw std::runtime_error("TAG3 scalar size mismatch");
        return flat;
    }
    if (r.starts_with("FIXD", 4)) {
        std::vector<std::int64_t> one = decode_fixed_stream(r);
        if (one.size() != expected_n) throw std::runtime_error("FIXD scalar length mismatch");
        return one;
    }
    if (r.starts_with("RANS1", 5)) {
        std::vector<std::int64_t> one = decode_rans1_stream(r);
        if (one.size() != expected_n) throw std::runtime_error("RANS scalar length mismatch");
        return one;
    }
    if (r.starts_with("BANS", 4) || r.starts_with("UANS", 4)) {
        return decode_bans_stream_cpu_unavailable(r);
    }
    throw std::runtime_error("unknown scalar component encoding");
}

struct ComponentPayload {
    std::string name;
    std::vector<unsigned char> bytes;
};

static ComponentPayload read_component(Reader& r, const char* expected_name) {
    std::string name = r.read_magic4();
    if (name != std::string(expected_name, 4)) {
        throw std::runtime_error("unexpected component name: expected " + std::string(expected_name, 4) + ", got " + name);
    }
    std::uint64_t len64 = r.read_le<std::uint64_t>();
    if (len64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("component too large");
    }
    return ComponentPayload{name, r.read_bytes(static_cast<std::size_t>(len64))};
}

// Self-contained decoder for the current compressor's LCP2 format.
// This code has no dependency on the compressor source or Python module.
// LCG1 decoding below remains available through the same public entry point.
namespace lcps_lcp2 {

constexpr std::uint32_t kMaxValueRansAlphabetCpp = 1U << 18;

struct BlockWrapParametersCpp {
    std::int64_t min_value = 0;
    std::int64_t max_value = 0;
    std::int64_t max_difference = 1;
    std::int64_t min_correction = 0;
    std::int64_t max_correction = 0;
};

static BlockWrapParametersCpp make_block_wrap_parameters_cpp(
    std::int64_t min_value,
    std::int64_t max_value
) {
    if (min_value > max_value) {
        throw std::runtime_error("invalid block-ID wrap range");
    }

    std::uint64_t difference = 0;
    if (min_value < 0 && max_value >= 0) {
        const std::uint64_t negative_magnitude =
            static_cast<std::uint64_t>(-(min_value + 1)) + 1;
        difference = static_cast<std::uint64_t>(max_value) +
                     negative_magnitude;
    } else {
        difference = static_cast<std::uint64_t>(max_value - min_value);
    }
    if (difference >=
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("block-ID range is too wide for wrapped differences");
    }

    BlockWrapParametersCpp result;
    result.min_value = min_value;
    result.max_value = max_value;
    result.max_difference = static_cast<std::int64_t>(difference + 1);
    result.max_correction = result.max_difference / 2;
    result.min_correction = -result.max_correction;
    if ((result.max_difference & 1) == 0) {
        --result.max_correction;
    }
    return result;
}

// Shared by encoder and decoder. The LCP2 format reconstructs float64 values
// with one correctly rounded FMA; do not replace it with reassociated arithmetic.
static double reconstruct_coordinate_cpp(
    std::int64_t q, double minimum, double eb
) {
    return std::fma(static_cast<double>(q) + 0.5, 2.0 * eb, minimum);
}

static void require_lcps_cpp(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static std::size_t checked_size_cpp(std::uint64_t value) {
    require_lcps_cpp(value <= std::numeric_limits<std::size_t>::max(),
                     "LCPS size does not fit size_t");
    return static_cast<std::size_t>(value);
}

static std::size_t checked_product_cpp(std::size_t a, std::size_t b) {
    require_lcps_cpp(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b,
                     "LCPS size overflow");
    return a * b;
}

class LcpsReaderCpp {
public:
    explicit LcpsReaderCpp(std::string_view bytes) : remaining_(bytes) {}

    std::string_view Bytes(std::size_t size) {
        require_lcps_cpp(size <= remaining_.size(), "truncated LCPS bitstream");
        const auto result = remaining_.substr(0, size);
        remaining_.remove_prefix(size);
        return result;
    }
    std::uint8_t U8() {
        return static_cast<std::uint8_t>(Bytes(1)[0]);
    }
    std::uint32_t U32() {
        std::uint32_t result = 0;
        for (int i = 0; i < 4; ++i) {
            result |= static_cast<std::uint32_t>(U8()) << (8 * i);
        }
        return result;
    }
    std::uint64_t U64() {
        std::uint64_t result = 0;
        for (int i = 0; i < 8; ++i) {
            result |= static_cast<std::uint64_t>(U8()) << (8 * i);
        }
        return result;
    }
    std::int64_t I64() {
        const auto bits = U64();
        std::int64_t result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
    double Double() {
        const auto bits = U64();
        double result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
    std::uint64_t VarUint() {
        std::uint64_t result = 0;
        for (int shift = 0; shift <= 63; shift += 7) {
            const auto byte = U8();
            require_lcps_cpp(shift != 63 || (byte & 0xfeU) == 0,
                             "LCPS varuint overflow");
            result |= static_cast<std::uint64_t>(byte & 0x7fU) << shift;
            if ((byte & 0x80U) == 0) {
                return result;
            }
        }
        throw std::runtime_error("invalid LCPS varuint");
    }
    std::size_t Size() { return checked_size_cpp(U64()); }
    std::string_view Blob() {
        const auto size = Size();
        return Bytes(size);
    }
    std::size_t Remaining() const { return remaining_.size(); }
    void End() const {
        require_lcps_cpp(remaining_.empty(), "unexpected trailing LCPS bytes");
    }
private:
    std::string_view remaining_;
};

static std::string zstd_decompress_cpp(std::string_view bytes, std::size_t size) {
    std::string result(size, '\0');
    const auto decoded = ZSTD_decompress(
        result.data(), result.size(), bytes.data(), bytes.size());
    if (ZSTD_isError(decoded)) {
        throw std::runtime_error(
            std::string("ZSTD_decompress failed: ") + ZSTD_getErrorName(decoded));
    }
    require_lcps_cpp(decoded == size, "LCPS Zstd output size mismatch");
    return result;
}

static std::vector<std::uint64_t> decode_normalized_value_rans_cpp(
    std::string_view bytes, std::size_t expected_count
) {
    LcpsReaderCpp reader(bytes);
    require_lcps_cpp(reader.Bytes(4) == "RVN1", "invalid RVN1 header");
    const int precision_bits = reader.U8();
    require_lcps_cpp(reader.Size() == expected_count, "rANS symbol count mismatch");
    std::uint64_t state = reader.U32();
    const auto table_bytes = reader.Blob();
    const auto renormalization = reader.Blob();
    reader.End();
    if (expected_count == 0) {
        require_lcps_cpp(precision_bits == 0 && state == 1 && table_bytes.empty()
                         && renormalization.empty(), "invalid empty rANS stream");
        return {};
    }
    require_lcps_cpp(precision_bits >= 12 && precision_bits <= 20,
                     "invalid rANS precision");
    const std::uint32_t precision = 1U << precision_bits;
    const std::uint32_t lower_bound = precision * 4U;
    require_lcps_cpp(state >= lower_bound && state < std::uint64_t(lower_bound) * 256,
                     "invalid rANS state");

    LcpsReaderCpp table_reader(table_bytes);
    const auto alphabet = checked_size_cpp(table_reader.VarUint());
    require_lcps_cpp(alphabet > 0 && alphabet <= kMaxValueRansAlphabetCpp,
                     "invalid rANS alphabet");
    std::vector<std::uint32_t> probabilities(alphabet, 0);
    for (std::size_t i = 0; i < alphabet;) {
        const auto first = table_reader.U8();
        const int token = first & 3U;
        if (token == 3) {
            const std::size_t zeros = (first >> 2) + 1;
            require_lcps_cpp(zeros <= alphabet - i, "invalid rANS zero run");
            i += zeros;
        } else {
            std::uint32_t probability = first >> 2;
            for (int b = 0; b < token; ++b) {
                probability |= static_cast<std::uint32_t>(table_reader.U8())
                               << (6 + 8 * b);
            }
            require_lcps_cpp(probability > 0, "invalid rANS probability");
            probabilities[i++] = probability;
        }
    }
    table_reader.End();

    std::vector<std::uint32_t> cumulative(alphabet, 0);
    std::vector<std::uint32_t> lookup(precision);
    std::uint32_t sum = 0;
    for (std::size_t symbol = 0; symbol < alphabet; ++symbol) {
        cumulative[symbol] = sum;
        require_lcps_cpp(probabilities[symbol] <= precision - sum,
                         "invalid rANS probability sum");
        const auto end = sum + probabilities[symbol];
        std::fill(lookup.begin() + sum, lookup.begin() + end,
                  static_cast<std::uint32_t>(symbol));
        sum = end;
    }
    require_lcps_cpp(sum == precision, "incomplete rANS probability table");

    LcpsReaderCpp renorm_reader(renormalization);
    std::vector<std::uint64_t> result(expected_count);
    for (auto& value : result) {
        const auto slot = static_cast<std::uint32_t>(state) & (precision - 1);
        const auto symbol = lookup[slot];
        value = symbol;
        state = std::uint64_t(probabilities[symbol]) * (state >> precision_bits)
              + slot - cumulative[symbol];
        while (state < lower_bound) {
            state = (state << 8) | renorm_reader.U8();
        }
    }
    renorm_reader.End();
    require_lcps_cpp(state == lower_bound, "rANS final state mismatch");
    return result;
}

static std::uint64_t read_msb_bits_cpp(
    std::string_view bytes, std::size_t& position, int width
) {
    require_lcps_cpp(width >= 0 && width <= 64, "invalid packed bit width");
    std::uint64_t value = 0;
    for (int bit = 0; bit < width; ++bit) {
        require_lcps_cpp(position / 8 < bytes.size(), "truncated packed values");
        value = (value << 1)
              | ((static_cast<std::uint8_t>(bytes[position / 8])
                  >> (7 - position % 8)) & 1U);
        ++position;
    }
    return value;
}

using LcpsIntegerMatrixCpp = std::vector<std::vector<std::int64_t>>;

static LcpsIntegerMatrixCpp decode_component_matrix_cpp(
    std::string_view bytes, std::size_t expected_dims, std::size_t max_records
) {
    LcpsReaderCpp reader(bytes);
    const auto magic = reader.Bytes(4);
    const std::size_t dims = reader.U32();
    const std::size_t n = reader.Size();
    require_lcps_cpp(dims == expected_dims && n <= max_records,
                     "LCPS component shape mismatch");
    const auto num_values = checked_product_cpp(dims, n);
    LcpsIntegerMatrixCpp result(dims, std::vector<std::int64_t>(n));

    if (magic == "RVMX") {
        const auto values = decode_normalized_value_rans_cpp(reader.Blob(), num_values);
        reader.End();
        std::size_t index = 0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t d = 0; d < dims; ++d) {
                result[d][i] = static_cast<std::int64_t>(values[index++]);
            }
        }
        return result;
    }

    if (magic == "ZFMX") {
        struct Dimension { std::int64_t minimum; int width; std::size_t bytes; };
        std::vector<Dimension> dimensions(dims);
        std::size_t total_bytes = 0;
        for (auto& dimension : dimensions) {
            dimension.minimum = reader.I64();
            dimension.width = reader.U8();
            dimension.bytes = reader.Size();
            require_lcps_cpp(dimension.width >= 1 && dimension.width <= 64,
                             "invalid fixed width");
            const auto bits = checked_product_cpp(n, dimension.width);
            require_lcps_cpp(dimension.bytes == bits / 8 + (bits % 8 != 0),
                             "invalid fixed payload size");
            require_lcps_cpp(dimension.bytes <=
                             std::numeric_limits<std::size_t>::max() - total_bytes,
                             "fixed payload size overflow");
            total_bytes += dimension.bytes;
        }
        require_lcps_cpp(reader.Size() == total_bytes, "fixed raw size mismatch");
        const auto raw = zstd_decompress_cpp(reader.Blob(), total_bytes);
        reader.End();
        LcpsReaderCpp packed_reader(raw);
        for (std::size_t d = 0; d < dims; ++d) {
            const auto& dimension = dimensions[d];
            const auto packed = packed_reader.Bytes(dimension.bytes);
            std::size_t position = 0;
            for (auto& value : result[d]) {
                const auto offset = read_msb_bits_cpp(packed, position, dimension.width);
                const auto maximum_offset =
                    static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                    - static_cast<std::uint64_t>(dimension.minimum);
                require_lcps_cpp(offset <= maximum_offset, "fixed value overflow");
                const auto bits = static_cast<std::uint64_t>(dimension.minimum) + offset;
                std::memcpy(&value, &bits, sizeof(value));
            }
        }
        packed_reader.End();
        return result;
    }

    require_lcps_cpp(magic == "ZTG4" || magic == "RTG4", "unknown LCPS encoding");
    const int maximum_width = reader.U8();
    const auto tags = decode_normalized_value_rans_cpp(reader.Blob(), n);
    std::size_t total_bits = 0;
    int actual_maximum_width = 0;
    for (const auto tag : tags) {
        require_lcps_cpp(tag >= 1 && tag <= 63, "invalid Tagged width");
        actual_maximum_width = std::max(actual_maximum_width, static_cast<int>(tag));
        const auto bits = checked_product_cpp(dims, static_cast<std::size_t>(tag));
        require_lcps_cpp(bits <= std::numeric_limits<std::size_t>::max() - total_bits,
                         "Tagged size overflow");
        total_bits += bits;
    }
    require_lcps_cpp(maximum_width == actual_maximum_width, "Tagged maximum width mismatch");
    const auto raw_size = total_bits / 8 + (total_bits % 8 != 0);
    std::string unpacked;
    std::string_view packed;
    if (magic == "ZTG4") {
        require_lcps_cpp(reader.Size() == raw_size, "Tagged raw size mismatch");
        unpacked = zstd_decompress_cpp(reader.Blob(), raw_size);
        packed = unpacked;
    } else {
        packed = reader.Blob();
        require_lcps_cpp(packed.size() == raw_size, "Tagged payload size mismatch");
    }
    reader.End();
    std::size_t position = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t d = 0; d < dims; ++d) {
            result[d][i] = static_cast<std::int64_t>(
                read_msb_bits_cpp(packed, position, static_cast<int>(tags[i])));
        }
    }
    return result;
}

struct LcpsDecodedPointsCpp {
    std::size_t n = 0;
    std::size_t dims = 0;
    std::vector<double> values;  // Point-major (N, D), float64.
};

static LcpsDecodedPointsCpp decode_lcps_bitstream_cpp(std::string_view bytes) {
    // LCP2, all numeric fields little-endian, no C++ struct padding:
    //   magic[4], uint32 D, uint64 N, int64 p, float64 eb,
    //   int64 wrap_min, int64 wrap_max, float64 mins[D],
    //   five (uint64 byte_length, encoded_matrix) pairs, in this order:
    //       idrel, diff, keep, choice2, rel,
    //   uint64 exact_count, uint64 exact_zstd_bytes, exact_zstd.
    // Exact corrections are Zstd((uint64 point_major_index, float64 value)*count).
    // An empty correction section has count=0, byte_length=0 and no Zstd frame.
    require_lcps_cpp(std::fegetround() == FE_TONEAREST,
                     "LCP2 decoding requires round-to-nearest floating-point mode");
    LcpsReaderCpp reader(bytes);
    require_lcps_cpp(reader.Bytes(4) == "LCP2", "expected a self-contained LCP2 bitstream");
    LcpsDecodedPointsCpp output;
    output.dims = reader.U32();
    output.n = reader.Size();
    const auto dims = output.dims;
    const auto n = output.n;
    require_lcps_cpp(dims > 0 && n > 0, "invalid LCP2 shape");
    const auto num_values = checked_product_cpp(n, dims);
    const auto p = reader.I64();
    const double eb = reader.Double();
    require_lcps_cpp(p > 0 && eb > 0.0 && std::isfinite(2.0 * eb),
                     "invalid LCP2 quantization parameters");
    const auto wrap_min = reader.I64();
    const auto wrap_max = reader.I64();
    require_lcps_cpp(wrap_min >= 0, "invalid LCP2 block range");
    const auto wrap = make_block_wrap_parameters_cpp(wrap_min, wrap_max);
    require_lcps_cpp(dims <= reader.Remaining() / sizeof(double), "truncated LCP2 minima");
    std::vector<double> minima(dims);
    for (auto& minimum : minima) {
        minimum = reader.Double();
        require_lcps_cpp(std::isfinite(minimum), "invalid LCP2 coordinate minimum");
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    const auto idrel = decode_component_matrix_cpp(reader.Blob(), 1, n);
    const auto differences = decode_component_matrix_cpp(reader.Blob(), dims, n);
    const auto kept = decode_component_matrix_cpp(reader.Blob(), dims, n);
    const auto choices = decode_component_matrix_cpp(reader.Blob(), 1, n);
    auto quantized = decode_component_matrix_cpp(reader.Blob(), dims, n);
    const auto exact_count = reader.Size();
    const auto exact_bytes = reader.Blob();
    reader.End();
    require_lcps_cpp(idrel[0].size() == n && quantized[0].size() == n,
                     "LCP2 point count mismatch");
    require_lcps_cpp(exact_count <= num_values, "invalid LCP2 correction count");
    auto t1 = std::chrono::high_resolution_clock::now();

    const auto compact_n = choices[0].size();
    LcpsIntegerMatrixCpp compact(dims, std::vector<std::int64_t>(compact_n));
    std::size_t diff_index = 0;
    std::size_t keep_index = 0;
    const auto range = static_cast<std::uint64_t>(wrap.max_difference);
    for (std::size_t i = 0; i < compact_n; ++i) {
        const auto choice = choices[0][i];
        require_lcps_cpp(choice == 0 || choice == 1, "invalid choice2 value");
        if (choice == 0) {
            require_lcps_cpp(keep_index < kept[0].size(), "missing kept block");
            for (std::size_t d = 0; d < dims; ++d) {
                const auto value = kept[d][keep_index];
                require_lcps_cpp(value >= wrap_min && value <= wrap_max,
                                 "kept block outside wrap range");
                compact[d][i] = value;
            }
            ++keep_index;
            continue;
        }
        require_lcps_cpp(i > 0 && diff_index < differences[0].size(),
                         "missing block difference or initial block");
        for (std::size_t d = 0; d < dims; ++d) {
            const auto signed_symbol = differences[d][diff_index];
            require_lcps_cpp(signed_symbol >= 0, "negative ZigZag symbol");
            const auto symbol = static_cast<std::uint64_t>(signed_symbol);
            const std::int64_t correction = (symbol & 1U)
                ? -static_cast<std::int64_t>(symbol >> 1) - 1
                : static_cast<std::int64_t>(symbol >> 1);
            require_lcps_cpp(correction >= wrap.min_correction && correction <= wrap.max_correction,
                             "block correction outside wrap range");
            const auto previous = static_cast<std::uint64_t>(compact[d][i - 1] - wrap_min);
            std::uint64_t offset;
            if (correction >= 0) {
                const auto amount = static_cast<std::uint64_t>(correction);
                offset = previous >= amount ? previous - amount : range - (amount - previous);
            } else {
                const auto amount = static_cast<std::uint64_t>(-(correction + 1)) + 1;
                offset = previous >= range - amount ? previous - (range - amount) : previous + amount;
            }
            compact[d][i] = wrap_min + static_cast<std::int64_t>(offset);
        }
        ++diff_index;
    }
    require_lcps_cpp(diff_index == differences[0].size() && keep_index == kept[0].size(),
                     "unused block differences or kept blocks");
    auto t2 = std::chrono::high_resolution_clock::now();

    // Process forward references backward. Convert offsets to q in place:
    // an already reconstructed successor's block is q[successor] / p.
    std::size_t compact_index = compact_n;
    for (std::size_t i = n; i-- > 0;) {
        const auto jump = idrel[0][i];
        require_lcps_cpp(jump >= 0, "negative successor distance");
        std::size_t successor = i;
        if (jump == 0) {
            require_lcps_cpp(compact_index > 0, "missing explicit block");
            --compact_index;
        } else {
            require_lcps_cpp(static_cast<std::uint64_t>(jump) < n - i,
                             "successor outside point array");
            successor += static_cast<std::size_t>(jump);
        }
        for (std::size_t d = 0; d < dims; ++d) {
            const auto offset = quantized[d][i];
            require_lcps_cpp(offset >= 0 && offset < p, "invalid block offset");
            const auto block = jump == 0 ? compact[d][compact_index]
                                        : quantized[d][successor] / p;
            require_lcps_cpp(block <= (std::numeric_limits<std::int64_t>::max() - offset) / p,
                             "quantized coordinate overflow");
            quantized[d][i] = block * p + offset;
        }
    }
    require_lcps_cpp(compact_index == 0, "unused explicit blocks");
    auto t3 = std::chrono::high_resolution_clock::now();

    output.values.resize(num_values);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t d = 0; d < dims; ++d) {
            output.values[i * dims + d] = reconstruct_coordinate_cpp(quantized[d][i], minima[d], eb);
        }
    }
    if (exact_count == 0) {
        require_lcps_cpp(exact_bytes.empty(), "unexpected exact correction bytes");
    } else {
        const auto raw = zstd_decompress_cpp(exact_bytes, checked_product_cpp(exact_count, 16));
        LcpsReaderCpp corrections(raw);
        std::size_t previous_index = 0;
        for (std::size_t i = 0; i < exact_count; ++i) {
            const auto index = corrections.Size();
            const double value = corrections.Double();
            require_lcps_cpp(index < num_values && (i == 0 || index > previous_index)
                             && std::isfinite(value), "invalid exact coordinate correction");
            output.values[index] = value;
            previous_index = index;
        }
        corrections.End();
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    double decoding = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
    double l2 = std::chrono::duration<double, std::milli>(t2 - t1).count() / 1000.0;
    double l1 = std::chrono::duration<double, std::milli>(t3 - t2).count() / 1000.0;
    double other = std::chrono::duration<double, std::milli>(t4 - t3).count() / 1000.0;
    printf("Decoding: %f\n", decoding);
    printf("level-2: %f\n", l2);
    printf("level-1: %f\n", l1);
    printf("dequantization+deblock: %f\n", other);
    return output;
}

static py::array_t<double> decode_python(std::string_view bytes, bool output_soa) {
    auto decoded = std::make_unique<LcpsDecodedPointsCpp>(decode_lcps_bitstream_cpp(bytes));
    const auto ssize_max = static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max());
    require_lcps_cpp(decoded->n <= ssize_max && decoded->dims <= ssize_max
                     && decoded->values.size() <= ssize_max / sizeof(double),
                     "decoded array exceeds Python shape limits");
    if (output_soa) {
        // Keep the old interface's C-contiguous (D, N) layout. Corrections have
        // already been applied using their point-major indices.
        py::array_t<double> result(std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(decoded->dims),
            static_cast<py::ssize_t>(decoded->n)
        });
        double* target = result.mutable_data();
        for (std::size_t d = 0; d < decoded->dims; ++d) {
            for (std::size_t i = 0; i < decoded->n; ++i) {
                target[d * decoded->n + i] = decoded->values[i * decoded->dims + d];
            }
        }
        return result;
    }

    // Transfer ownership of the point-major vector to NumPy without copying.
    auto* storage = decoded.get();
    py::capsule owner(storage, [](void* pointer) {
        delete static_cast<LcpsDecodedPointsCpp*>(pointer);
    });
    decoded.release();
    return py::array_t<double>(
        {static_cast<py::ssize_t>(storage->n), static_cast<py::ssize_t>(storage->dims)},
        {static_cast<py::ssize_t>(storage->dims * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        storage->values.data(), owner);
}

}  // namespace lcps_lcp2

static py::array_t<double> lcps_decompress_serial(py::bytes compressed, bool output_soa = false) {
    std::string s = compressed;
    if (s.size() >= 4 && s.compare(0, 4, "LCP2") == 0) {
        return lcps_lcp2::decode_python(s, output_soa);
    }
    Reader r(reinterpret_cast<const unsigned char*>(s.data()), s.size());

    r.expect_magic("LCG1", 4);
    const std::uint32_t version = r.read_le<std::uint32_t>();
    if (version != 1u) throw std::runtime_error("unsupported LCG1 version");
    const std::uint32_t D_u32 = r.read_le<std::uint32_t>();
    const std::uint64_t N_u64 = r.read_le<std::uint64_t>();
    const std::int64_t p = r.read_le<std::int64_t>();
    const std::int64_t threshold = r.read_le<std::int64_t>();
    const std::int64_t diff = r.read_le<std::int64_t>();
    (void)threshold;
    (void)diff;
    const double eb = r.read_le<double>();
    const std::uint64_t num_block_u64 = r.read_le<std::uint64_t>();
    const std::uint64_t num_idrel_u64 = r.read_le<std::uint64_t>();

    if (D_u32 == 0) throw std::runtime_error("invalid D in header");
    if (D_u32 > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) throw std::runtime_error("D too large");
    if (N_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        num_block_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        num_idrel_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("header sizes too large");
    }
    const int D = static_cast<int>(D_u32);
    const std::size_t N = static_cast<std::size_t>(N_u64);
    const std::size_t num_block = static_cast<std::size_t>(num_block_u64);
    const std::size_t num_idrel = static_cast<std::size_t>(num_idrel_u64);

    if (N == 0 || p <= 0 || !std::isfinite(eb) || eb <= 0.0) {
        throw std::runtime_error("invalid LCG1 scalar metadata");
    }
    if (num_block > N || num_idrel > N || num_block != N - num_idrel) {
        throw std::runtime_error("LCG1 compact-stream counts do not sum to N");
    }
    (void)checked_mul_size(static_cast<std::size_t>(D), N, "LCG1 D*N");
    if (N > static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max())
        || static_cast<std::size_t>(D) > static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max())) {
        throw std::runtime_error("LCG1 output shape exceeds Python index range");
    }

    std::vector<double> mins(D);
    for (int d = 0; d < D; ++d) {
        mins[d] = r.read_le<double>();
        if (!std::isfinite(mins[d])) throw std::runtime_error("non-finite LCG1 minimum");
    }

    ComponentPayload c_blk = read_component(r, "BLK0");
    ComponentPayload c_idr = read_component(r, "IDR0");
    ComponentPayload c_rel = read_component(r, "REL0");
    ComponentPayload c_cho = read_component(r, "CHO0");
    if (r.remaining() != 0) throw std::runtime_error("trailing bytes after LCG1 container");

    std::vector<unsigned char> blk_raw = maybe_zstd_decompress_component_payload(c_blk.bytes);
    std::vector<unsigned char> idr_raw = maybe_zstd_decompress_component_payload(c_idr.bytes);
    std::vector<unsigned char> rel_raw = maybe_zstd_decompress_component_payload(c_rel.bytes);
    std::vector<unsigned char> cho_raw = maybe_zstd_decompress_component_payload(c_cho.bytes);

    Reader rb(blk_raw.data(), blk_raw.size());
    std::vector<std::int64_t> outBlock = decode_matrix_component_payload(rb, D, num_block);
    if (rb.remaining() != 0) throw std::runtime_error("BLK0 payload trailing bytes");

    Reader ri(idr_raw.data(), idr_raw.size());
    std::vector<std::int64_t> idrel = decode_scalar_component_payload(ri, num_idrel);
    if (ri.remaining() != 0) throw std::runtime_error("IDR0 payload trailing bytes");

    Reader rr(rel_raw.data(), rel_raw.size());
    std::vector<std::int64_t> rel = decode_matrix_component_payload(rr, D, N);
    if (rr.remaining() != 0) throw std::runtime_error("REL0 payload trailing bytes");

    Reader rc(cho_raw.data(), cho_raw.size());
    std::vector<std::int64_t> choice64 = decode_scalar_component_payload(rc, N);
    if (rc.remaining() != 0) throw std::runtime_error("CHO0 payload trailing bytes");

    std::vector<unsigned char> choice(N);
    std::size_t ones = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (choice64[i] != 0 && choice64[i] != 1) throw std::runtime_error("choice stream is not binary");
        choice[i] = static_cast<unsigned char>(choice64[i]);
        if (choice[i]) ++ones;
    }
    if (ones != num_idrel) throw std::runtime_error("num_idrel does not match number of choice=1 entries");
    if (N - ones != num_block) throw std::runtime_error("num_block does not match number of choice=0 entries");

    std::vector<std::int64_t> bid(static_cast<std::size_t>(D) * N, 0);
    std::size_t block_pos = num_block;
    std::size_t idrel_pos = num_idrel;
    for (std::size_t rev = 0; rev < N; ++rev) {
        const std::size_t i = N - 1 - rev;
        if (choice[i] == 0) {
            if (block_pos == 0) throw std::runtime_error("outBlock underflow during reconstruction");
            --block_pos;
            for (int d = 0; d < D; ++d) {
                bid[static_cast<std::size_t>(d) * N + i] = outBlock[static_cast<std::size_t>(d) * num_block + block_pos];
            }
        } else {
            if (idrel_pos == 0) throw std::runtime_error("IDRel underflow during reconstruction");
            --idrel_pos;
            const std::int64_t rid = idrel[idrel_pos];
            if (rid <= 0) throw std::runtime_error("invalid non-positive relID");
            const std::uint64_t next64 = static_cast<std::uint64_t>(i) + static_cast<std::uint64_t>(rid);
            if (next64 >= N) throw std::runtime_error("relID points outside particle array");
            const std::size_t next = static_cast<std::size_t>(next64);
            for (int d = 0; d < D; ++d) {
                bid[static_cast<std::size_t>(d) * N + i] = bid[static_cast<std::size_t>(d) * N + next];
            }
        }
    }
    if (block_pos != 0 || idrel_pos != 0) throw std::runtime_error("not all outBlock/IDRel entries consumed");

    const double scale = 2.0 * eb;
    if (output_soa) {
        py::array_t<double> arr(std::vector<py::ssize_t>{static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(N)});
        auto m = arr.mutable_unchecked<2>();
        for (int d = 0; d < D; ++d) {
            for (std::size_t i = 0; i < N; ++i) {
                const std::int64_t q = bid[static_cast<std::size_t>(d) * N + i] * p + rel[static_cast<std::size_t>(d) * N + i];
                m(d, static_cast<py::ssize_t>(i)) = mins[d] + (static_cast<double>(q) + 0.5) * scale;
            }
        }
        return arr;
    } else {
        py::array_t<double> arr(std::vector<py::ssize_t>{static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(D)});
        auto m = arr.mutable_unchecked<2>();
        for (std::size_t i = 0; i < N; ++i) {
            for (int d = 0; d < D; ++d) {
                const std::int64_t q = bid[static_cast<std::size_t>(d) * N + i] * p + rel[static_cast<std::size_t>(d) * N + i];
                m(static_cast<py::ssize_t>(i), d) = mins[d] + (static_cast<double>(q) + 0.5) * scale;
            }
        }
        return arr;
    }
}

PYBIND11_MODULE(lcps_cpu_decompress, m) {
    m.doc() = "Independent serial CPU decompressor for self-contained LCP2 and legacy LCG1 streams.";
    m.def("lcps_decompress_serial", &lcps_decompress_serial,
          py::arg("compressed"), py::arg("output_soa") = false,
          "Decode compressor result[0] into float64 coordinates in original order. "
          "output_soa=False returns (N, D); True returns C-contiguous (D, N). "
          "LCP2 metadata, codebooks and exact coordinate corrections are read from compressed.");
}
