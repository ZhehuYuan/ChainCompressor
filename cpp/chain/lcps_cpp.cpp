#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>
#include <zstd.h>

namespace py = pybind11;

#if defined(__FAST_MATH__)
#error "LCPS error-bound checks require compilation without -ffast-math"
#endif


static inline std::uint64_t zorder_part1by2_64(std::uint64_t n) {
    n &= 0x1FFFFFULL;  // keep low 21 bits

    n = (n | (n << 32)) & 0x1F00000000FFFFULL;
    n = (n | (n << 16)) & 0x1F0000FF0000FFULL;
    n = (n | (n << 8))  & 0x100F00F00F00F00FULL;
    n = (n | (n << 4))  & 0x10C30C30C30C30C3ULL;
    n = (n | (n << 2))  & 0x1249249249249249ULL;

    return n;
}

static inline std::int64_t flatten_block_id(std::int64_t bx,
                                            std::int64_t by,
                                            std::int64_t bz) {
    if (bx < 0 || by < 0 || bz < 0) {
        throw std::runtime_error(
            "Z-order encoding requires non-negative block coordinates"
        );
    }

    constexpr std::int64_t LIMIT = static_cast<std::int64_t>(1) << 21;

    if (bx >= LIMIT || by >= LIMIT || bz >= LIMIT) {
        throw std::runtime_error(
            "Z-order encoder supports coordinates < 2^21 per axis"
        );
    }

    std::uint64_t ux = static_cast<std::uint64_t>(bx);
    std::uint64_t uy = static_cast<std::uint64_t>(by);
    std::uint64_t uz = static_cast<std::uint64_t>(bz);

    std::uint64_t code =
        zorder_part1by2_64(ux)
        | (zorder_part1by2_64(uy) << 1)
        | (zorder_part1by2_64(uz) << 2);

    return static_cast<std::int64_t>(code);
}

static inline void append_u64_le(std::string& out, std::uint64_t x) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>((x >> (8 * i)) & 0xff));
    }
}

static inline void append_u32_le(std::string& out, std::uint32_t x) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((x >> (8 * i)) & 0xff));
    }
}

static inline void append_u16_le(std::string& out, std::uint16_t x) {
    for (int i = 0; i < 2; ++i) {
        out.push_back(static_cast<char>((x >> (8 * i)) & 0xff));
    }
}

static inline void append_varuint(std::string& out, std::uint64_t value) {
    while (value >= 0x80U) {
        out.push_back(static_cast<char>(
            static_cast<std::uint8_t>(value) | 0x80U
        ));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

static inline void append_i64_le(std::string& out, std::int64_t x) {
    append_u64_le(out, static_cast<std::uint64_t>(x));
}

static inline void append_double_le(std::string& out, double x) {
    static_assert(sizeof(double) == 8, "double must be 8 bytes");
    std::uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(double));
    append_u64_le(out, u);
}


static std::string zstd_compress_cpp(const std::string& raw, int level = 22) {
    const std::size_t bound = ZSTD_compressBound(raw.size());
    std::string out(bound, '\0');

    const std::size_t compressed_size = ZSTD_compress(
        out.data(),
        out.size(),
        raw.data(),
        raw.size(),
        level
    );

    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error(
            std::string("ZSTD_compress failed: ") + ZSTD_getErrorName(compressed_size)
        );
    }

    out.resize(compressed_size);
    return out;
}

static inline int unsigned_width_like_python(const std::vector<std::int64_t>& values,
                                             std::int64_t& mn_out) {
    if (values.empty()) {
        mn_out = 0;
        return 1;
    }

    auto [mn_it, mx_it] = std::minmax_element(values.begin(), values.end());

    std::int64_t mn = *mn_it;
    std::int64_t mx = *mx_it;

    mn_out = mn;

    if (mn == mx) {
        return 1;
    }

    std::uint64_t range = static_cast<std::uint64_t>(mx)
                        - static_cast<std::uint64_t>(mn);
    int width = 1;
    while ((range >>= 1) != 0) {
        ++width;
    }
    return width;
}

static std::string pack_unsigned_like_python(const std::vector<std::int64_t>& values,
                                             std::int64_t mn,
                                             int width) {
    if (width <= 0) {
        return std::string();
    }

    // Python int has arbitrary precision, but MSVC does not support __uint128_t.
    // This routine writes bits directly into bytes, MSB first, which matches:
    //     acc = (acc << width) | (v & mask)
    //     while acc_bits >= 8: output the highest 8 bits
    // For int64 input values, width cannot practically exceed 64.
    if (width > 64) {
        throw std::runtime_error("pack_unsigned_like_python only supports width <= 64 on MSVC");
    }

    std::string out;
    out.reserve((values.size() * static_cast<std::size_t>(width) + 7) / 8);

    std::uint64_t mask = std::numeric_limits<std::uint64_t>::max();
    if (width < 64) {
        mask = (std::uint64_t(1) << width) - 1;
    }

    unsigned char current_byte = 0;
    int filled_bits = 0;

    for (std::int64_t raw_v : values) {
        if (raw_v < mn) {
            throw std::runtime_error("_pack_unsigned expects non-negative values");
        }

        // Use unsigned modular subtraction to avoid signed overflow when
        // mn is very negative and raw_v is very positive.
        std::uint64_t v =
            static_cast<std::uint64_t>(raw_v) - static_cast<std::uint64_t>(mn);
        if (v > mask) {
            throw std::runtime_error("fixed value does not fit in its bit width");
        }

        // Append exactly `width` bits, from high bit to low bit.
        for (int bit = width - 1; bit >= 0; --bit) {
            current_byte = static_cast<unsigned char>(
                (current_byte << 1) | ((v >> bit) & 1ULL)
            );
            ++filled_bits;

            if (filled_bits == 8) {
                out.push_back(static_cast<char>(current_byte));
                current_byte = 0;
                filled_bits = 0;
            }
        }
    }

    if (filled_bits > 0) {
        current_byte = static_cast<unsigned char>(current_byte << (8 - filled_bits));
        out.push_back(static_cast<char>(current_byte));
    }

    return out;
}

static std::string fixed_length_encode_like_python(
    const std::vector<std::int64_t>& arr
) {
    std::int64_t mn = 0;
    int width = unsigned_width_like_python(arr, mn);

    std::string out;

    // b"FIXD"
    out.push_back('F');
    out.push_back('I');
    out.push_back('X');
    out.push_back('D');

    // struct.pack("<qQB", mn, arr.size, width)
    append_i64_le(out, mn);
    append_u64_le(out, static_cast<std::uint64_t>(arr.size()));
    out.push_back(static_cast<char>(width & 0xff));

    // payload
    std::string payload = pack_unsigned_like_python(arr, mn, width);
    // std::string payload = pack_unsigned_like_python(arr, mn, std::max(32, width));
    out += payload;

    return out;
}


// ============================================================
// Huffman coding (canonical), matching LCPSStepReorder.py::huffman_encode
// ============================================================

struct HuffNodeCpp {
    std::uint64_t freq = 0;
    bool is_leaf = false;
    std::int64_t symbol = 0;
    int left = -1;
    int right = -1;
};

struct HuffHeapItemCpp {
    std::uint64_t freq = 0;
    std::size_t seq = 0;
    int node = -1;
};

struct HuffHeapGreaterCpp {
    bool operator()(const HuffHeapItemCpp& a, const HuffHeapItemCpp& b) const {
        if (a.freq != b.freq) {
            return a.freq > b.freq;
        }
        return a.seq > b.seq;
    }
};

static std::unordered_map<std::int64_t, int> build_huffman_code_lengths_cpp(
    const std::vector<std::int64_t>& arr
) {
    std::unordered_map<std::int64_t, std::size_t> first_index;
    std::vector<std::int64_t> symbols;
    std::vector<std::uint64_t> freqs;

    for (std::int64_t v : arr) {
        auto it = first_index.find(v);
        if (it == first_index.end()) {
            std::size_t idx = symbols.size();
            first_index.emplace(v, idx);
            symbols.push_back(v);
            freqs.push_back(1);
        } else {
            ++freqs[it->second];
        }
    }

    std::unordered_map<std::int64_t, int> lengths;
    if (symbols.empty()) {
        return lengths;
    }

    if (symbols.size() == 1) {
        lengths[symbols[0]] = 1;
        return lengths;
    }

    std::vector<HuffNodeCpp> nodes;
    nodes.reserve(symbols.size() * 2);

    std::priority_queue<
        HuffHeapItemCpp,
        std::vector<HuffHeapItemCpp>,
        HuffHeapGreaterCpp
    > heap;

    std::size_t seq = 0;
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        HuffNodeCpp node;
        node.freq = freqs[i];
        node.is_leaf = true;
        node.symbol = symbols[i];
        nodes.push_back(node);
        heap.push(HuffHeapItemCpp{freqs[i], seq, static_cast<int>(nodes.size() - 1)});
        ++seq;
    }

    while (heap.size() > 1) {
        HuffHeapItemCpp h1 = heap.top();
        heap.pop();
        HuffHeapItemCpp h2 = heap.top();
        heap.pop();

        HuffNodeCpp parent;
        parent.freq = h1.freq + h2.freq;
        parent.is_leaf = false;
        parent.left = h1.node;
        parent.right = h2.node;
        nodes.push_back(parent);

        heap.push(HuffHeapItemCpp{
            parent.freq,
            seq,
            static_cast<int>(nodes.size() - 1)
        });
        ++seq;
    }

    int root = heap.top().node;

    std::vector<std::pair<int, int>> stack;
    stack.emplace_back(root, 0);

    while (!stack.empty()) {
        auto [node_idx, depth] = stack.back();
        stack.pop_back();

        const HuffNodeCpp& node = nodes[node_idx];
        if (node.is_leaf) {
            lengths[node.symbol] = std::max(depth, 1);
        } else {
            // Python DFS visits left before right. Order does not affect lengths,
            // but pushing right first preserves the same traversal order.
            stack.emplace_back(node.right, depth + 1);
            stack.emplace_back(node.left, depth + 1);
        }
    }

    return lengths;
}

static std::string huffman_encode_like_python(const std::vector<std::int64_t>& arr) {
    std::unordered_map<std::int64_t, int> lengths =
        build_huffman_code_lengths_cpp(arr);

    std::vector<std::pair<std::int64_t, int>> symbols_sorted;
    symbols_sorted.reserve(lengths.size());
    for (const auto& kv : lengths) {
        symbols_sorted.emplace_back(kv.first, kv.second);
    }

    std::sort(
        symbols_sorted.begin(),
        symbols_sorted.end(),
        [](const auto& a, const auto& b) {
            if (a.second != b.second) {
                return a.second < b.second;
            }
            return a.first < b.first;
        }
    );

    struct CodeCpp {
        std::uint64_t code = 0;
        int bitlen = 0;
    };

    std::unordered_map<std::int64_t, CodeCpp> codes;
    codes.reserve(symbols_sorted.size() * 2 + 1);

    std::uint64_t code = 0;
    int prev_len = symbols_sorted.empty() ? 0 : symbols_sorted[0].second;

    for (const auto& item : symbols_sorted) {
        std::int64_t sym = item.first;
        int bitlen = item.second;
        if (bitlen < prev_len) {
            throw std::runtime_error("internal Huffman error: code lengths not sorted");
        }
        if (bitlen - prev_len >= 64) {
            throw std::runtime_error("Huffman code length shift exceeds 63 bits");
        }
        code <<= (bitlen - prev_len);
        if (bitlen > 64) {
            throw std::runtime_error("huffman_encode_like_python only supports code lengths <= 64");
        }
        codes.emplace(sym, CodeCpp{code, bitlen});
        ++code;
        prev_len = bitlen;
    }

    std::string packed;
    std::uint64_t bit_count = 0;
    unsigned char current_byte = 0;
    int filled_bits = 0;

    for (std::int64_t v : arr) {
        auto it = codes.find(v);
        if (it == codes.end()) {
            throw std::runtime_error("internal Huffman error: missing code");
        }
        std::uint64_t c = it->second.code;
        int bitlen = it->second.bitlen;
        bit_count += static_cast<std::uint64_t>(bitlen);

        for (int bit = bitlen - 1; bit >= 0; --bit) {
            current_byte = static_cast<unsigned char>(
                (current_byte << 1) | ((c >> bit) & 1ULL)
            );
            ++filled_bits;
            if (filled_bits == 8) {
                packed.push_back(static_cast<char>(current_byte));
                current_byte = 0;
                filled_bits = 0;
            }
        }
    }

    if (filled_bits > 0) {
        current_byte = static_cast<unsigned char>(current_byte << (8 - filled_bits));
        packed.push_back(static_cast<char>(current_byte));
    }

    std::string out;
    out.push_back('H');
    out.push_back('U');
    out.push_back('F');
    out.push_back('F');

    // struct.pack("<QI Q", arr.size, len(symbols_sorted), bit_count)
    append_u64_le(out, static_cast<std::uint64_t>(arr.size()));
    append_u32_le(out, static_cast<std::uint32_t>(symbols_sorted.size()));
    append_u64_le(out, bit_count);

    // for sym, bitlen in symbols_sorted: struct.pack("<qH", sym, bitlen)
    for (const auto& item : symbols_sorted) {
        append_i64_le(out, item.first);
        append_u16_le(out, static_cast<std::uint16_t>(item.second));
    }

    out += packed;
    return out;
}

static int bit_length_u64(std::uint64_t v) {
    int bits = 0;
    while (v != 0) {
        ++bits;
        v >>= 1;
    }
    return bits;
}

static std::string pack_variable_unsigned_like_python(
    const std::vector<std::uint64_t>& values,
    const std::vector<int>& widths
) {
    if (values.size() != widths.size()) {
        throw std::runtime_error("values and widths must have the same length");
    }

    std::string out;
    unsigned char current_byte = 0;
    int filled_bits = 0;

    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint64_t v = values[i];
        int w = widths[i];

        if (w < 0) {
            throw std::runtime_error("width must be non-negative");
        }

        if (w == 0) {
            if (v != 0) {
                throw std::runtime_error("width is 0 but value is not 0");
            }
            continue;
        }

        if (w < 64) {
            if (v >= (std::uint64_t(1) << w)) {
                throw std::runtime_error("value does not fit in width bits");
            }
        }

        for (int bit = w - 1; bit >= 0; --bit) {
            current_byte = static_cast<unsigned char>(
                (current_byte << 1) | ((v >> bit) & 1ULL)
            );
            ++filled_bits;

            if (filled_bits == 8) {
                out.push_back(static_cast<char>(current_byte));
                current_byte = 0;
                filled_bits = 0;
            }
        }
    }

    if (filled_bits > 0) {
        current_byte = static_cast<unsigned char>(current_byte << (8 - filled_bits));
        out.push_back(static_cast<char>(current_byte));
    }

    return out;
}

constexpr std::uint32_t kMaxValueRansAlphabetCpp = 1U << 18;

static int level10_rans_precision_bits_cpp(std::size_t num_unique_symbols) {
    if (num_unique_symbols == 0) {
        throw std::runtime_error("cannot choose rANS precision for no symbols");
    }
    const int unique_symbol_bits = std::min(
        18,
        bit_length_u64(static_cast<std::uint64_t>(num_unique_symbols)) + 2
    );
    return std::min(
        20, std::max(12, (3 * unique_symbol_bits) / 2)
    );
}

struct NormalizedValueRansTableCpp {
    int precision_bits = 12;
    std::vector<std::uint32_t> probabilities;
    std::vector<std::uint32_t> cumulative;
};

static NormalizedValueRansTableCpp build_normalized_value_rans_table_cpp(
    const std::vector<std::uint64_t>& values
) {
    if (values.empty()) {
        throw std::runtime_error("cannot build an rANS table for an empty stream");
    }

    const std::uint64_t max_value =
        *std::max_element(values.begin(), values.end());
    if (max_value >= kMaxValueRansAlphabetCpp) {
        throw std::runtime_error(
            "Raw value-rANS alphabet exceeds the 18-bit limit"
        );
    }

    std::vector<std::uint64_t> frequencies(
        static_cast<std::size_t>(max_value) + 1, 0
    );
    for (std::uint64_t value : values) {
        ++frequencies[static_cast<std::size_t>(value)];
    }

    std::size_t num_unique_symbols = 0;
    for (std::uint64_t frequency : frequencies) {
        num_unique_symbols += frequency != 0;
    }

    NormalizedValueRansTableCpp table;
    table.precision_bits =
        level10_rans_precision_bits_cpp(num_unique_symbols);
    const std::uint32_t precision = 1U << table.precision_bits;
    const long double total = static_cast<long double>(values.size());

    table.probabilities.resize(frequencies.size(), 0);
    std::int64_t probability_sum = 0;
    for (std::size_t i = 0; i < frequencies.size(); ++i) {
        const std::uint64_t frequency = frequencies[i];
        std::uint32_t probability = static_cast<std::uint32_t>(
            std::floor(
                static_cast<long double>(frequency) * precision / total
                + 0.5L
            )
        );
        if (probability == 0 && frequency != 0) {
            probability = 1;
        }
        table.probabilities[i] = probability;
        probability_sum += probability;
    }

    if (probability_sum != precision) {
        std::vector<std::size_t> sorted(table.probabilities.size());
        std::iota(sorted.begin(), sorted.end(), 0);
        std::stable_sort(
            sorted.begin(), sorted.end(),
            [&](std::size_t lhs, std::size_t rhs) {
                return table.probabilities[lhs] <
                       table.probabilities[rhs];
            }
        );

        if (probability_sum < precision) {
            table.probabilities[sorted.back()] +=
                static_cast<std::uint32_t>(precision - probability_sum);
            probability_sum = precision;
        } else {
            std::int64_t error = probability_sum - precision;
            while (error > 0) {
                const long double relative =
                    static_cast<long double>(precision) /
                    static_cast<long double>(probability_sum);
                bool changed = false;

                for (std::size_t j = sorted.size(); j-- > 1 && error > 0;) {
                    std::uint32_t& probability =
                        table.probabilities[sorted[j]];
                    if (probability <= 1) {
                        if (j == sorted.size() - 1) {
                            throw std::runtime_error(
                                "rANS normalization cannot retain every symbol"
                            );
                        }
                        break;
                    }

                    const std::uint32_t new_probability =
                        static_cast<std::uint32_t>(
                            std::floor(relative * probability)
                        );
                    std::int64_t reduction =
                        static_cast<std::int64_t>(probability) -
                        new_probability;
                    if (reduction == 0) {
                        reduction = 1;
                    }
                    if (reduction >= probability) {
                        reduction = probability - 1;
                    }
                    reduction = std::min(reduction, error);
                    probability -= static_cast<std::uint32_t>(reduction);
                    probability_sum -= reduction;
                    error -= reduction;
                    changed = true;
                }

                if (!changed) {
                    throw std::runtime_error(
                        "rANS normalization did not converge"
                    );
                }
            }
        }
    }

    table.cumulative.resize(table.probabilities.size(), 0);
    std::uint32_t cumulative = 0;
    for (std::size_t i = 0; i < table.probabilities.size(); ++i) {
        table.cumulative[i] = cumulative;
        cumulative += table.probabilities[i];
    }
    if (cumulative != precision || probability_sum != precision) {
        throw std::runtime_error("normalized rANS probability total mismatch");
    }
    return table;
}

static std::string encode_compact_probability_table_cpp(
    const std::vector<std::uint32_t>& probabilities
) {
    if (probabilities.empty() ||
        probabilities.size() > kMaxValueRansAlphabetCpp) {
        throw std::runtime_error("invalid normalized rANS alphabet size");
    }

    std::string out;
    append_varuint(out, static_cast<std::uint64_t>(probabilities.size()));
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        const std::uint32_t probability = probabilities[i];
        if (probability == 0) {
            std::uint32_t additional_zeros = 0;
            while (additional_zeros < 63 &&
                   i + additional_zeros + 1 < probabilities.size() &&
                   probabilities[i + additional_zeros + 1] == 0) {
                ++additional_zeros;
            }
            out.push_back(static_cast<char>(
                (additional_zeros << 2) | 3U
            ));
            i += additional_zeros;
            continue;
        }

        int extra_bytes = 0;
        if (probability >= (1U << 6)) {
            ++extra_bytes;
            if (probability >= (1U << 14)) {
                ++extra_bytes;
            }
        }
        out.push_back(static_cast<char>(
            (probability << 2) | static_cast<std::uint32_t>(extra_bytes)
        ));
        for (int b = 0; b < extra_bytes; ++b) {
            out.push_back(static_cast<char>(
                probability >> (8 * (b + 1) - 2)
            ));
        }
    }
    return out;
}

struct NormalizedValueRansPartsCpp {
    std::uint64_t num_values = 0;
    int precision_bits = 0;
    std::uint32_t final_state = 1;
    std::string probability_table;
    std::string renormalization;
};

static NormalizedValueRansPartsCpp build_normalized_value_rans_parts_cpp(
    const std::vector<std::uint64_t>& values
) {
    NormalizedValueRansPartsCpp parts;
    parts.num_values = static_cast<std::uint64_t>(values.size());
    if (values.empty()) {
        return parts;
    }

    const NormalizedValueRansTableCpp table =
        build_normalized_value_rans_table_cpp(values);
    parts.precision_bits = table.precision_bits;
    parts.probability_table =
        encode_compact_probability_table_cpp(table.probabilities);

    const std::uint32_t precision = 1U << table.precision_bits;
    const std::uint32_t lower_bound = precision * 4U;
    std::uint64_t state = lower_bound;

    for (auto it = values.rbegin(); it != values.rend(); ++it) {
        const std::size_t symbol = static_cast<std::size_t>(*it);
        const std::uint64_t probability = table.probabilities[symbol];
        const std::uint64_t start = table.cumulative[symbol];
        const std::uint64_t x_max =
            (static_cast<std::uint64_t>(lower_bound) >>
             table.precision_bits) * 256U * probability;

        while (state >= x_max) {
            parts.renormalization.push_back(
                static_cast<char>(state & 0xffU)
            );
            state >>= 8;
        }

        state = (state / probability) * precision
              + (state % probability) + start;
        if (state >= static_cast<std::uint64_t>(lower_bound) * 256U) {
            throw std::runtime_error("normalized value-rANS state overflow");
        }
    }

    std::reverse(
        parts.renormalization.begin(), parts.renormalization.end()
    );
    parts.final_state = static_cast<std::uint32_t>(state);
    return parts;
}

static std::string normalized_value_rans_encode_cpp(
    const std::vector<std::uint64_t>& values
) {
    const NormalizedValueRansPartsCpp parts =
        build_normalized_value_rans_parts_cpp(values);

    // Normalized value-rANS framing:
    //   "RVN1"
    //   uint8  precision_bits (0 for an empty stream)
    //   uint64 number_of_values
    //   uint32 final_state
    //   uint64 compact_probability_table_bytes
    //   compact table: varuint alphabet size, encoded probabilities and
    //                  run-length tokens for consecutive zero probabilities
    //   uint64 renormalization_bytes
    //   renormalization bytes
    std::string out;
    out.reserve(
        std::size_t(4 + 1 + 8 + 4 + 8 + 8)
        + parts.probability_table.size() + parts.renormalization.size()
    );
    out.append("RVN1", 4);
    out.push_back(static_cast<char>(parts.precision_bits & 0xff));
    append_u64_le(out, parts.num_values);
    append_u32_le(out, parts.final_state);
    append_u64_le(
        out, static_cast<std::uint64_t>(parts.probability_table.size())
    );
    out += parts.probability_table;
    append_u64_le(
        out, static_cast<std::uint64_t>(parts.renormalization.size())
    );
    out += parts.renormalization;
    return out;
}

static std::string fixed_stream_external_header_zstd_cpp(
    const std::vector<std::int64_t>& stream
) {
    std::int64_t mn = 0;
    const int width = unsigned_width_like_python(stream, mn);
    auto t1 = std::chrono::high_resolution_clock::now();
    const std::string payload = pack_unsigned_like_python(stream, mn, width);
    auto t2 = std::chrono::high_resolution_clock::now();
    const std::string compressed_payload = zstd_compress_cpp(payload);
    auto t3 = std::chrono::high_resolution_clock::now();
    
    double s;
    s = std::chrono::duration<double, std::milli>(t2 - t1).count() / 1000.0;
    //printf("Fix: %f\n", s);
    s = std::chrono::duration<double, std::milli>(t3 - t2).count() / 1000.0;
    //printf("Fix ZSTD: %f\n", s);

    // Format mirrors the GPU-side ZFXD framing conceptually: all metadata is
    // outside Zstd, and only the bit-packed payload is compressed.
    std::string out;
    out.reserve(
        std::size_t(4 + 8 + 8 + 1 + 8 + 8) + compressed_payload.size()
    );
    out.append("ZFXD", 4);
    append_i64_le(out, mn);
    append_u64_le(out, static_cast<std::uint64_t>(stream.size()));
    out.push_back(static_cast<char>(width & 0xff));
    append_u64_le(out, static_cast<std::uint64_t>(payload.size()));
    append_u64_le(out, static_cast<std::uint64_t>(compressed_payload.size()));
    out += compressed_payload;
    return out;
}

static std::string fixed_matrix_external_header_zstd_cpp(
    const std::vector<std::vector<std::int64_t>>& streams
) {
    if (streams.empty()) {
        return std::string("EMPTY", 5);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    const std::size_t dims = streams.size();
    const std::size_t n = streams[0].size();
    for (std::size_t d = 1; d < dims; ++d) {
        if (streams[d].size() != n) {
            throw std::runtime_error("all fixed streams must have the same length");
        }
    }
    if (dims > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("too many dimensions for ZFMX");
    }

    struct FixedDimensionCpp {
        std::int64_t mn = 0;
        int width = 1;
        std::string payload;
    };

    std::vector<FixedDimensionCpp> encoded(dims);
    std::size_t total_payload_bytes = 0;
    for (std::size_t d = 0; d < dims; ++d) {
        auto& one = encoded[d];
        one.width = unsigned_width_like_python(streams[d], one.mn);
        one.payload = pack_unsigned_like_python(streams[d], one.mn, one.width);
        if (one.payload.size() >
            std::numeric_limits<std::size_t>::max() - total_payload_bytes) {
            throw std::runtime_error("fixed matrix payload size overflow");
        }
        total_payload_bytes += one.payload.size();
    }

    std::string concatenated_payload;
    concatenated_payload.reserve(total_payload_bytes);
    for (const auto& one : encoded) {
        concatenated_payload += one.payload;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    const std::string compressed_payload = zstd_compress_cpp(concatenated_payload);
    auto t3 = std::chrono::high_resolution_clock::now();
    
    double s;
    s = std::chrono::duration<double, std::milli>(t2 - t1).count() / 1000.0;
    //printf("Fix: %f\n", s);
    s = std::chrono::duration<double, std::milli>(t3 - t2).count() / 1000.0;
    //printf("Fix Zstd: %f\n", s);

    // Same outer-metadata policy as the GPU ZFMX path:
    //   "ZFMX" + D + n
    //   repeated D times: mn + width + raw_payload_bytes
    //   total_raw_payload_bytes + zstd_payload_bytes
    //   Zstd([packed dim0][packed dim1]...)
    std::string out;
    out.reserve(
        std::size_t(4 + 4 + 8)
        + dims * std::size_t(8 + 1 + 8)
        + std::size_t(8 + 8)
        + compressed_payload.size()
    );
    out.append("ZFMX", 4);
    append_u32_le(out, static_cast<std::uint32_t>(dims));
    append_u64_le(out, static_cast<std::uint64_t>(n));
    for (const auto& one : encoded) {
        append_i64_le(out, one.mn);
        out.push_back(static_cast<char>(one.width & 0xff));
        append_u64_le(out, static_cast<std::uint64_t>(one.payload.size()));
    }
    append_u64_le(out, static_cast<std::uint64_t>(total_payload_bytes));
    append_u64_le(out, static_cast<std::uint64_t>(compressed_payload.size()));
    out += compressed_payload;
    return out;
}

static bool normalized_raw_rans_matrix_cpp(
    const std::vector<std::vector<std::int64_t>>& streams,
    std::string& encoded_out
) {
    if (streams.empty()) {
        encoded_out = std::string("EMPTY", 5);
        return true;
    }

    const std::size_t dims = streams.size();
    const std::size_t n = streams[0].size();
    for (std::size_t d = 1; d < dims; ++d) {
        if (streams[d].size() != n) {
            throw std::runtime_error(
                "all Raw rANS streams must have the same length"
            );
        }
    }
    if (dims > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("too many dimensions for RVMX");
    }
    if (n != 0 && dims > std::numeric_limits<std::size_t>::max() / n) {
        throw std::runtime_error("Raw rANS matrix size overflow");
    }

    // Feed integer values to rANS directly in point-major order. No value is
    // first converted into a byte stream or a bit-packed stream.
    std::vector<std::uint64_t> point_major;
    point_major.reserve(n * dims);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t d = 0; d < dims; ++d) {
            const std::int64_t value = streams[d][i];
            if (value < 0) {
                throw std::runtime_error(
                    "Raw rANS expects non-negative integer values"
                );
            }
            const std::uint64_t symbol =
                static_cast<std::uint64_t>(value);
            if (symbol >= kMaxValueRansAlphabetCpp) {
                return false;
            }
            point_major.push_back(symbol);
        }
    }

    const std::string rans_payload =
        normalized_value_rans_encode_cpp(point_major);

    //   "RVMX" + uint32 D + uint64 n + uint64 rans_bytes
    //   + RVN1(point-major integer values)
    std::string out;
    out.reserve(std::size_t(4 + 4 + 8 + 8) + rans_payload.size());
    out.append("RVMX", 4);
    append_u32_le(out, static_cast<std::uint32_t>(dims));
    append_u64_le(out, static_cast<std::uint64_t>(n));
    append_u64_le(out, static_cast<std::uint64_t>(rans_payload.size()));
    out += rans_payload;
    encoded_out = std::move(out);
    return true;
}

struct NormalizedTaggedMatrixPartsCpp {
    std::size_t dims = 0;
    std::size_t n = 0;
    int max_value_width = 0;
    std::string tag_payload;
    std::string packed_values;
};

static NormalizedTaggedMatrixPartsCpp build_normalized_tagged_matrix_parts_cpp(
    const std::vector<std::vector<std::int64_t>>& streams
) {
    NormalizedTaggedMatrixPartsCpp parts;
    parts.dims = streams.size();
    parts.n = streams[0].size();
    for (std::size_t d = 1; d < parts.dims; ++d) {
        if (streams[d].size() != parts.n) {
            throw std::runtime_error(
                "all Tagged streams must have the same length"
            );
        }
    }
    if (parts.dims > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("too many dimensions for Tagged encoding");
    }
    if (parts.n != 0 &&
        parts.dims > std::numeric_limits<std::size_t>::max() / parts.n) {
        throw std::runtime_error("Tagged matrix size overflow");
    }

    std::vector<std::uint64_t> tags;
    std::vector<std::uint64_t> flat_values;
    std::vector<int> flat_widths;
    tags.reserve(parts.n);
    flat_values.reserve(parts.n * parts.dims);
    flat_widths.reserve(parts.n * parts.dims);

    for (std::size_t i = 0; i < parts.n; ++i) {
        int tag = 1;
        for (std::size_t d = 0; d < parts.dims; ++d) {
            const std::int64_t value = streams[d][i];
            if (value < 0) {
                throw std::runtime_error(
                    "Tagged encoding expects non-negative integer values"
                );
            }
            tag = std::max(
                tag, bit_length_u64(static_cast<std::uint64_t>(value))
            );
        }
        tags.push_back(static_cast<std::uint64_t>(tag));
        parts.max_value_width = std::max(parts.max_value_width, tag);

        for (std::size_t d = 0; d < parts.dims; ++d) {
            flat_values.push_back(
                static_cast<std::uint64_t>(streams[d][i])
            );
            flat_widths.push_back(tag);
        }
    }

    parts.tag_payload = normalized_value_rans_encode_cpp(tags);
    parts.packed_values =
        pack_variable_unsigned_like_python(flat_values, flat_widths);
    return parts;
}

static std::string normalized_tagged_matrix_cpp(
    const std::vector<std::vector<std::int64_t>>& streams
) {
    if (streams.empty()) {
        return std::string("EMPTY", 5);
    }

    const NormalizedTaggedMatrixPartsCpp parts =
        build_normalized_tagged_matrix_parts_cpp(streams);

    // Tags use normalized value-rANS; packed values are emitted directly.
    //   "RTG4" + uint32 D + uint64 n + uint8 max_value_width
    //   + uint64 tag_rans_bytes + RVN1(tags)
    //   + uint64 packed_value_bytes + packed_values
    std::string out;
    out.reserve(
        std::size_t(4 + 4 + 8 + 1 + 8 + 8)
        + parts.tag_payload.size() + parts.packed_values.size()
    );
    out.append("RTG4", 4);
    append_u32_le(out, static_cast<std::uint32_t>(parts.dims));
    append_u64_le(out, static_cast<std::uint64_t>(parts.n));
    out.push_back(static_cast<char>(parts.max_value_width & 0xff));
    append_u64_le(out, static_cast<std::uint64_t>(parts.tag_payload.size()));
    out += parts.tag_payload;
    append_u64_le(out, static_cast<std::uint64_t>(parts.packed_values.size()));
    out += parts.packed_values;
    return out;
}

static std::string tagged_scheme_encode_matrix_separate_zstd_cpp(
    const std::vector<std::vector<std::int64_t>>& streams
) {
    // Final CPU tagged format:
    //   "ZTG4"
    //   uint32 D
    //   uint64 n
    //   uint8  max_value_width
    //   uint64 tag_payload_bytes
    //   tag_payload = normalized RVN1 wrapper; tags never pass through Zstd
    //   uint64 value_raw_bytes
    //   uint64 value_zstd_bytes
    //   value_zstd = Zstd(bit-packed point-major values)
    // Every custom header is outside Zstd.
    if (streams.empty()) {
        return std::string("EMPTY", 5);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const NormalizedTaggedMatrixPartsCpp parts =
        build_normalized_tagged_matrix_parts_cpp(streams);
    auto t2 = std::chrono::high_resolution_clock::now();
    const std::string value_zstd = zstd_compress_cpp(parts.packed_values);
    auto t3 = std::chrono::high_resolution_clock::now();

    double s;
    s = std::chrono::duration<double, std::milli>(t2 - t1).count() / 1000.0;
    //printf("Tag: %f\n", s);
    s = std::chrono::duration<double, std::milli>(t3 - t2).count() / 1000.0;
    //printf("Tag ZSTD: %f\n", s);

    std::string out;
    out.reserve(
        std::size_t(4 + 4 + 8 + 1 + 8 + 8 + 8)
        + parts.tag_payload.size() + value_zstd.size()
    );
    out.append("ZTG4", 4);
    append_u32_le(out, static_cast<std::uint32_t>(parts.dims));
    append_u64_le(out, static_cast<std::uint64_t>(parts.n));
    out.push_back(static_cast<char>(parts.max_value_width & 0xff));
    append_u64_le(out, static_cast<std::uint64_t>(parts.tag_payload.size()));
    out += parts.tag_payload;
    append_u64_le(out, static_cast<std::uint64_t>(parts.packed_values.size()));
    append_u64_le(out, static_cast<std::uint64_t>(value_zstd.size()));
    out += value_zstd;
    return out;
}

static std::string choose_and_encode_component_bytes_matrix_cpp(
    const std::vector<std::vector<std::int64_t>>& streams
) {
    if (streams.empty()) {
        return std::string("EMPTY", 5);
    }

    // Candidate and tie order:
    //   1. Tagged + Zstd (normalized value-rANS tags, Zstd-packed values)
    //   2. Fixed + Zstd
    //   3. Raw + normalized value-rANS, when the alphabet fits 18 bits
    //   4. Tagged (normalized value-rANS tags, directly emitted packed values)
    std::string best =
	//fixed_matrix_external_header_zstd_cpp(streams);
        tagged_scheme_encode_matrix_separate_zstd_cpp(streams);
    const char* best_name = "tag+zstd";
    //const char* best_name = "fix+zstd";

    auto consider = [&](std::string candidate, const char* name) {
        if (candidate.size() < best.size()) {
            best = std::move(candidate);
            best_name = name;
        }
    };

    consider(fixed_matrix_external_header_zstd_cpp(streams), "fix+zstd");

    std::string raw_candidate;
    if (normalized_raw_rans_matrix_cpp(streams, raw_candidate)) {
        consider(std::move(raw_candidate), "raw+rANS");
    }
    //consider(normalized_tagged_matrix_cpp(streams), "tag");
    printf("%s\n", best_name);
    return best;
}

static std::size_t choose_and_encode_component_size_matrix_cpp(
    const std::vector<std::vector<std::int64_t>>& streams
) {
    return choose_and_encode_component_bytes_matrix_cpp(streams).size();
}

// Store a component verbatim when the Encoding stage is disabled.
// RAW1 + uint32 dimensions + uint64 records + point-major int64 values.
static std::string raw_component_matrix_cpp(
    const std::vector<std::vector<std::int64_t>>& streams
) {
    const std::size_t dims = streams.size();
    if (dims == 0 || dims > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid RAW1 dimension count");
    }
    const std::size_t n = streams[0].size();
    for (const auto& stream : streams) {
        if (stream.size() != n) {
            throw std::runtime_error("all RAW1 streams must have the same length");
        }
    }
    if (n > (std::numeric_limits<std::size_t>::max() - 16) / 8 / dims) {
        throw std::runtime_error("RAW1 matrix size overflow");
    }
    std::string out;
    out.reserve(16 + n * dims * 8);
    out.append("RAW1", 4);
    append_u32_le(out, static_cast<std::uint32_t>(dims));
    append_u64_le(out, static_cast<std::uint64_t>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t d = 0; d < dims; ++d) {
            append_i64_le(out, streams[d][i]);
        }
    }
    return out;
}

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

static std::uint64_t zigzag_encode_i64_cpp(std::int64_t value) {
    if (value >= 0) {
        return static_cast<std::uint64_t>(value) << 1;
    }
    const std::uint64_t magnitude_minus_one =
        static_cast<std::uint64_t>(-(value + 1));
    return (magnitude_minus_one << 1) | 1ULL;
}

static std::uint64_t checked_record_bit_cost_cpp(
    std::size_t dims,
    int value_width,
    int extra_bits = 0
) {
    if (value_width < 0 || extra_bits < 0) {
        throw std::runtime_error("negative bit width");
    }
    const std::uint64_t width = static_cast<std::uint64_t>(value_width);
    if (width != 0 &&
        dims > std::numeric_limits<std::uint64_t>::max() / width) {
        throw std::runtime_error("record bit-cost overflow");
    }
    const std::uint64_t value_bits =
        static_cast<std::uint64_t>(dims) * width;
    const std::uint64_t extra = static_cast<std::uint64_t>(extra_bits);
    if (value_bits > std::numeric_limits<std::uint64_t>::max() - extra) {
        throw std::runtime_error("record bit-cost overflow");
    }
    return value_bits + extra;
}

static std::uint64_t absolute_difference_nonnegative_cpp(
    std::int64_t lhs,
    std::int64_t rhs
) {
    if (lhs < 0 || rhs < 0) {
        throw std::runtime_error("block IDs must be non-negative");
    }
    const std::uint64_t a = static_cast<std::uint64_t>(lhs);
    const std::uint64_t b = static_cast<std::uint64_t>(rhs);
    return a >= b ? a - b : b - a;
}

struct DiffKeepChoice2StreamsCpp {
    std::vector<std::vector<std::int64_t>> diff_streams;
    std::vector<std::vector<std::int64_t>> keep_streams;
    std::vector<std::int64_t> choice2_stream;
    BlockWrapParametersCpp wrap;
};

static DiffKeepChoice2StreamsCpp build_diff_keep_choice2_streams_cpp(
    const std::vector<std::vector<std::int64_t>>& out_block_streams
) {
    constexpr std::uint64_t kRequiredSavingBits = 0;

    DiffKeepChoice2StreamsCpp result;
    const std::size_t dims = out_block_streams.size();
    if (dims == 0) {
        result.wrap = make_block_wrap_parameters_cpp(0, 0);
        return result;
    }

    const std::size_t num_records = out_block_streams[0].size();
    for (std::size_t d = 1; d < dims; ++d) {
        if (out_block_streams[d].size() != num_records) {
            throw std::runtime_error("all outBlockID streams must have the same length");
        }
    }

    result.diff_streams.resize(dims);
    result.keep_streams.resize(dims);
    result.choice2_stream.reserve(num_records);
    for (std::size_t d = 0; d < dims; ++d) {
        result.diff_streams[d].reserve(num_records);
        result.keep_streams[d].reserve(num_records);
    }

    if (num_records == 0) {
        result.wrap = make_block_wrap_parameters_cpp(0, 0);
        return result;
    }

    std::int64_t min_block_id = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_block_id = std::numeric_limits<std::int64_t>::min();
    std::uint64_t max_absolute_difference = 0;
    for (std::size_t i = 0; i < num_records; ++i) {
        for (std::size_t d = 0; d < dims; ++d) {
            const std::int64_t value = out_block_streams[d][i];
            if (value < 0) {
                throw std::runtime_error("block IDs must be non-negative");
            }
            min_block_id = std::min(min_block_id, value);
            max_block_id = std::max(max_block_id, value);
            if (i != 0) {
                max_absolute_difference = std::max(
                    max_absolute_difference,
                    absolute_difference_nonnegative_cpp(
                        out_block_streams[d][i - 1], value)
                );
            }
        }
    }

    result.wrap =
        make_block_wrap_parameters_cpp(min_block_id, max_block_id);

    const int fixed_value_width =
        bit_length_u64(static_cast<std::uint64_t>(max_block_id));
    const int fixed_difference_width =
        bit_length_u64(max_absolute_difference);
    const std::uint64_t fixed_value_bits =
        checked_record_bit_cost_cpp(dims, fixed_value_width);
    const std::uint64_t fixed_difference_bits =
        checked_record_bit_cost_cpp(dims, fixed_difference_width);

    for (std::size_t i = 0; i < num_records; ++i) {
        bool use_difference = false;
        if (i != 0) {
            std::uint64_t max_value = 0;
            std::uint64_t max_difference = 0;
            for (std::size_t d = 0; d < dims; ++d) {
                max_value = std::max(
                    max_value,
                    static_cast<std::uint64_t>(out_block_streams[d][i])
                );
                max_difference = std::max(
                    max_difference,
                    absolute_difference_nonnegative_cpp(
                        out_block_streams[d][i - 1],
                        out_block_streams[d][i]
                    )
                );
            }

            const int tagged_value_width = bit_length_u64(max_value);
            const int tagged_difference_width =
                bit_length_u64(max_difference);
            const std::uint64_t tagged_value_bits =
                checked_record_bit_cost_cpp(
                    dims,
                    tagged_value_width,
                    bit_length_u64(
                        static_cast<std::uint64_t>(tagged_value_width)
                    )
                );
            const std::uint64_t tagged_difference_bits =
                checked_record_bit_cost_cpp(
                    dims,
                    tagged_difference_width,
                    bit_length_u64(
                        static_cast<std::uint64_t>(tagged_difference_width)
                    )
                );

            const std::uint64_t value_bits =
                std::min(tagged_value_bits, fixed_value_bits);
            const std::uint64_t difference_bits =
                std::max(tagged_difference_bits, fixed_difference_bits);
            use_difference =
                value_bits >= difference_bits &&
                value_bits - difference_bits >= kRequiredSavingBits;
        }

        result.choice2_stream.push_back(use_difference ? 1 : 0);
        if (!use_difference) {
            for (std::size_t d = 0; d < dims; ++d) {
                result.keep_streams[d].push_back(out_block_streams[d][i]);
            }
            continue;
        }

        for (std::size_t d = 0; d < dims; ++d) {
            const std::int64_t prediction = std::min(
                result.wrap.max_value,
                std::max(
                    result.wrap.min_value,
                    out_block_streams[d][i - 1]
                )
            );
            std::int64_t correction =
                prediction - out_block_streams[d][i];
            if (correction < result.wrap.min_correction) {
                correction += result.wrap.max_difference;
            } else if (correction > result.wrap.max_correction) {
                correction -= result.wrap.max_difference;
            }

            const std::uint64_t symbol = zigzag_encode_i64_cpp(correction);
            if (symbol > static_cast<std::uint64_t>(
                             std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error("ZigZag block difference exceeds int64");
            }
            result.diff_streams[d].push_back(
                static_cast<std::int64_t>(symbol)
            );
        }
    }

    return result;
}

static DiffKeepChoice2StreamsCpp keep_all_blocks_cpp(
    const std::vector<std::vector<std::int64_t>>& blocks
) {
    DiffKeepChoice2StreamsCpp result;
    const std::size_t dims = blocks.size();
    const std::size_t n = blocks[0].size();
    result.diff_streams.resize(dims);
    result.keep_streams = blocks;
    result.choice2_stream.assign(n, 0);
    std::int64_t minimum = std::numeric_limits<std::int64_t>::max();
    std::int64_t maximum = 0;
    for (const auto& dimension : blocks) {
        if (dimension.size() != n) {
            throw std::runtime_error("inconsistent block stream lengths");
        }
        for (const auto value : dimension) {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }
    result.wrap = make_block_wrap_parameters_cpp(n == 0 ? 0 : minimum, maximum);
    return result;
}

struct VectorKeyHash {
    std::size_t operator()(const std::vector<std::int64_t>& v) const noexcept {
        std::size_t h = 1469598103934665603ull;
        for (std::int64_t x : v) {
            std::uint64_t y = static_cast<std::uint64_t>(x);
            for (int i = 0; i < 8; ++i) {
                h ^= static_cast<std::size_t>((y >> (8 * i)) & 0xffu);
                h *= 1099511628211ull;
            }
        }
        return h;
    }
};

// Shared by encoder and decoder. The LCP2 format reconstructs float64 values
// with one correctly rounded FMA; do not replace it with reassociated arithmetic.
static double reconstruct_coordinate_cpp(
    std::int64_t q, double minimum, double eb
) {
    return std::fma(static_cast<double>(q) + 0.5, 2.0 * eb, minimum);
}

static bool within_error_bound_cpp(double restored, double original, double eb) {
    if (!std::isfinite(restored)) {
        return false;
    }
    const double difference = restored - original;
    const double magnitude = std::abs(difference);
    if (magnitude < eb) {
        return true;
    }
    if (!(magnitude == eb)) {
        return false;
    }

    // TwoSum recovers the rounding remainder of restored + (-original).
    // At the boundary, a rounded difference alone could accept an exact
    // difference slightly larger than eb. All operations use round-to-nearest.
    const double virtual_original = difference - restored;
    const double remainder =
        (restored - (difference - virtual_original))
        + (-original - virtual_original);
    return difference >= 0.0 ? remainder <= 0.0 : remainder >= 0.0;
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

    if (magic == "RAW1") {
        require_lcps_cpp(reader.Remaining() == checked_product_cpp(num_values, 8),
                         "RAW1 matrix size mismatch");
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t d = 0; d < dims; ++d) {
                result[d][i] = reader.I64();
            }
        }
        reader.End();
        return result;
    }

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
    // LCP2 (encoded) or LCP3 (raw), all numeric fields little-endian:
    //   magic[4], uint32 D, uint64 N, int64 p, float64 eb,
    //   int64 wrap_min, int64 wrap_max, float64 mins[D],
    //   five (uint64 byte_length, encoded_matrix) pairs, in this order:
    //       idrel, diff, keep, choice2, rel,
    //   uint64 exact_count, uint64 exact_zstd_bytes, exact_zstd.
    // LCP2 exact corrections use Zstd; LCP3 stores the same records verbatim.
    // An empty correction section has count=0, byte_length=0 and no Zstd frame.
    require_lcps_cpp(std::fegetround() == FE_TONEAREST,
                     "LCP2 decoding requires round-to-nearest floating-point mode");
    LcpsReaderCpp reader(bytes);
    const auto magic = reader.Bytes(4);
    require_lcps_cpp(magic == "LCP2" || magic == "LCP3",
                     "expected a self-contained LCP2 or LCP3 bitstream");
    const bool raw_components = magic == "LCP3";
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
    auto read_component = [&](std::size_t expected_dims) {
        const auto component = reader.Blob();
        if (raw_components) {
            require_lcps_cpp(component.size() >= 4 && component.substr(0, 4) == "RAW1",
                             "LCP3 requires raw component streams");
        }
        return decode_component_matrix_cpp(component, expected_dims, n);
    };
    const auto idrel = read_component(1);
    const auto differences = read_component(dims);
    const auto kept = read_component(dims);
    const auto choices = read_component(1);
    auto quantized = read_component(dims);
    const auto exact_count = reader.Size();
    const auto exact_bytes = reader.Blob();
    reader.End();
    require_lcps_cpp(idrel[0].size() == n && quantized[0].size() == n,
                     "LCP2 point count mismatch");
    require_lcps_cpp(exact_count <= num_values, "invalid LCP2 correction count");

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

    output.values.resize(num_values);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t d = 0; d < dims; ++d) {
            output.values[i * dims + d] = reconstruct_coordinate_cpp(quantized[d][i], minima[d], eb);
        }
    }
    if (exact_count == 0) {
        require_lcps_cpp(exact_bytes.empty(), "unexpected exact correction bytes");
    } else {
        std::string decompressed;
        std::string_view corrections_bytes = exact_bytes;
        const auto raw_size = checked_product_cpp(exact_count, 16);
        if (raw_components) {
            require_lcps_cpp(exact_bytes.size() == raw_size,
                             "raw exact correction size mismatch");
        } else {
            decompressed = zstd_decompress_cpp(exact_bytes, raw_size);
            corrections_bytes = decompressed;
        }
        LcpsReaderCpp corrections(corrections_bytes);
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
    return output;
}

static py::array_t<double> lcps_decompress2_cpp(const py::bytes& bitstream) {
    const std::string bytes = bitstream;
    auto decoded = std::make_unique<LcpsDecodedPointsCpp>(decode_lcps_bitstream_cpp(bytes));
    const auto ssize_max = static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max());
    require_lcps_cpp(decoded->n <= ssize_max && decoded->dims <= ssize_max / sizeof(double),
                     "decoded array exceeds Python shape limits");
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



static py::list lcps_compress2_cpp(
    py::object points_obj,
    double eb,
    std::int64_t p,
    std::int64_t threshold_arg,
    std::int64_t diff_arg,
    bool bl,
    bool l1,
    bool l2,
    bool ec
) {
    require_lcps_cpp(std::fegetround() == FE_TONEAREST,
                     "LCP2 encoding requires round-to-nearest floating-point mode");
    if (!(eb > 0.0) || !std::isfinite(2.0 * eb)) {
        throw std::invalid_argument("eb must be positive with finite 2*eb");
    }

    if (p <= 0) {
        throw std::invalid_argument("p must be positive");
    }
    const std::int64_t effective_p = bl ? p : 1;

    auto pts =
        py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(points_obj);

    if (!pts) {
        throw std::invalid_argument("points cannot be converted to float64 ndarray");
    }

    py::buffer_info buf = pts.request();

    if (buf.ndim != 2) {
        throw std::invalid_argument("points must be a 2-D ndarray with shape (N, D)");
    }

    const std::size_t N = static_cast<std::size_t>(buf.shape[0]);
    const std::size_t dims = static_cast<std::size_t>(buf.shape[1]);

    if (N == 0) {
        throw std::runtime_error(
            "zero-size array to reduction operation minimum which has no identity"
        );
    }
    if (dims == 0) {
        throw std::invalid_argument("points must have at least one dimension");
    }
    require_lcps_cpp(dims <= std::numeric_limits<std::uint32_t>::max(),
                     "too many dimensions for LCP2");
    (void)checked_product_cpp(N, dims);

    auto t1 = std::chrono::high_resolution_clock::now();

    const double* data = static_cast<const double*>(buf.ptr);
    const double denom = 2.0 * eb;

    std::vector<double> mins(dims, std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t d = 0; d < dims; ++d) {
            require_lcps_cpp(std::isfinite(data[i * dims + d]),
                             "LCPS coordinates must be finite");
            mins[d] = std::min(mins[d], data[i * dims + d]);
        }
    }

    std::vector<std::vector<std::int64_t>> bid_streams(
        dims, std::vector<std::int64_t>(N)
    );
    std::vector<std::vector<std::int64_t>> rel_streams(
        dims, std::vector<std::int64_t>(N)
    );

    std::string exact_corrections;
    std::uint64_t exact_count = 0;
    int64_t max_bid_bits = 0;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t d = 0; d < dims; ++d) {
            const double original = data[i * dims + d];
            const double quantized_value = std::floor((original - mins[d]) / denom);
            require_lcps_cpp(quantized_value >= 0.0 && quantized_value < 0x1p63,
                             "quantized coordinate does not fit int64");
            const auto q = static_cast<std::int64_t>(quantized_value);
            const double restored = reconstruct_coordinate_cpp(q, mins[d], eb);
            if (!within_error_bound_cpp(restored, original, eb)) {
                append_u64_le(exact_corrections, static_cast<std::uint64_t>(i * dims + d));
                append_double_le(exact_corrections, original);
                ++exact_count;
            }
            std::int64_t b = q / effective_p;
            bid_streams[d][i] = b;
	    max_bid_bits = std::max(b, max_bid_bits);
            rel_streams[d][i] = q - b * effective_p;
        }
    }
    max_bid_bits = bit_length_u64(max_bid_bits);
    
    auto t2 = std::chrono::high_resolution_clock::now();

    // Record each point's next point in the same block while scanning in index
    // order. This replaces the later hash lookup and lower_bound search.
    std::vector<std::size_t> next_in_same_block(N, N);
    if (l1) {
        std::unordered_map<
            std::vector<std::int64_t>,
            std::size_t,
            VectorKeyHash
        > last_in_block;
        last_in_block.reserve(N * 2);
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<std::int64_t> key(dims);
            for (std::size_t d = 0; d < dims; ++d) {
                key[d] = bid_streams[d][i];
            }

            auto it = last_in_block.find(key);
            if (it == last_in_block.end()) {
                last_in_block.emplace(std::move(key), i);
            } else {
                next_in_same_block[it->second] = i;
                it->second = i;
            }
        }
    }

    std::vector<std::int64_t> outIDRel;
    std::vector<std::vector<std::int64_t>> outBlockStreams(
        dims, std::vector<std::int64_t>()
    );

    outIDRel.reserve(N);
    for (std::size_t d = 0; d < dims; ++d) {
        outBlockStreams[d].reserve(N);
    }
    

    //std::size_t maxDiff = 0;

    //int64_t maxBlockID[3] = {0, 0, 0};

    for (std::size_t i = 0; i < N; ++i) {
        if (next_in_same_block[i] != N) {
            std::size_t relID_size_t = next_in_same_block[i] - i;

            std::int64_t max_bid = bid_streams[0][i];
            for (std::size_t d = 1; d < dims; ++d) {
                max_bid = std::max(max_bid, bid_streams[d][i]);
            }

            int local_bits = bit_length_u64(static_cast<std::uint64_t>(max_bid));
	    int tag_bits = bit_length_u64(static_cast<std::uint64_t>(local_bits));
	    int smaller = std::min(static_cast<std::int64_t>(local_bits) * static_cast<std::int64_t>(dims) + tag_bits, static_cast<std::int64_t>(dims) * max_bid_bits);
            bool threshold_smaller_than_block_bits =
                bit_length_u64(static_cast<std::int64_t>(relID_size_t)) 
		+ bit_length_u64(bit_length_u64(static_cast<std::int64_t>(relID_size_t)))
                < 
		smaller 
		//static_cast<std::int64_t>(dims) * max_bid_bits
		- threshold_arg;

            /*bool relid_fits_threshold = true;
            if (threshold_arg >= 0) {
                if (threshold_arg >= 63) {
                    relid_fits_threshold = true;
                } else {
                    std::uint64_t relid_threshold =
                        std::uint64_t(1) << static_cast<unsigned>(threshold_arg);
                    relid_fits_threshold =
                        static_cast<std::uint64_t>(relID_size_t) < relid_threshold;
                }
            }*/

            //if (relid_fits_threshold && 
	    if (threshold_smaller_than_block_bits) {
                outIDRel.push_back(static_cast<std::int64_t>(relID_size_t));
		//maxDiff = std::max(maxDiff, relID_size_t);
                continue;
            }
        }

        outIDRel.push_back(0);
        for (std::size_t d = 0; d < dims; ++d) {
            outBlockStreams[d].push_back(bid_streams[d][i]);
	    //maxBlockID[d] = std::max(maxBlockID[d], bid_streams[d][i]);
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    //printf("maxDiff: %d\n", maxDiff);

    /*
    std::vector<std::int64_t> outBlockFlatten;
    for(int i = 0; i < outBlockStreams[0].size(); i++){
	int64_t tmp = 0;
	for(int j = 0; j < 3; j++){
		if(j>0) tmp *= maxBlockID[j-1];
		tmp += outBlockStreams[j][i];
	}
    	outBlockFlatten.push_back(tmp);
    }*/
    //printf("flatten size: %d\n", outBlockFlatten.size());
    //std::vector<std::vector<std::int64_t>> block_streams(1);
    //block_streams[0] = outBlockFlatten;
    DiffKeepChoice2StreamsCpp block_split = l2
        ? build_diff_keep_choice2_streams_cpp(outBlockStreams)
        : keep_all_blocks_cpp(outBlockStreams);
    auto t4 = std::chrono::high_resolution_clock::now();

    auto encode_component = [ec](const std::vector<std::vector<std::int64_t>>& streams) {
        return ec ? choose_and_encode_component_bytes_matrix_cpp(streams)
                  : raw_component_matrix_cpp(streams);
    };

    std::string diff_bytes = encode_component(
        block_split.diff_streams
    );
    std::string keep_bytes = encode_component(
        block_split.keep_streams
    );
    std::vector<std::vector<std::int64_t>> choice2_streams(1);
    choice2_streams[0] = std::move(block_split.choice2_stream);
    std::string choice2_bytes = encode_component(
        choice2_streams
    );

    const std::size_t block_replacement_size =
        diff_bytes.size() + keep_bytes.size() + choice2_bytes.size();

    std::vector<std::vector<std::int64_t>> idrel_streams(1);
    idrel_streams[0] = outIDRel;
    std::string idrel_bytes = encode_component(
        idrel_streams
    );

    std::string rel_bytes = encode_component(
        rel_streams
    );
    auto t5 = std::chrono::high_resolution_clock::now();
    
    double s;
    s = std::chrono::duration<double, std::milli>(t2 - t1).count() / 1000.0;
    printf("Quantization+block: %f\n", s);
    s = std::chrono::duration<double, std::milli>(t3 - t2).count() / 1000.0;
    printf("l1: %f\n", s);
    s = std::chrono::duration<double, std::milli>(t4 - t3).count() / 1000.0;
    printf("l2: %f\n", s);
    s = std::chrono::duration<double, std::milli>(t5 - t4).count() / 1000.0;
    printf("Encoding: %f\n", s);
    
    /*
    std::vector<std::vector<std::int64_t>> choice_streams(1);
    choice_streams[0] = outChoice;
    std::string choice_bytes = choose_and_encode_component_bytes_matrix_cpp(
        choice_streams
    );

    auto t5 = std::chrono::high_resolution_clock::now();

    std::string container;
    container.reserve(
        std::size_t(4 + 4 + 4 + 8 + 8 + 8 + 8 + 8 + 8 + 8) +
        dims * sizeof(double) +
        4 * (4 + 8) +
        block_bytes.size() + idrel_bytes.size() + rel_bytes.size() + choice_bytes.size()
    );

    container.append("LCG1", 4);
    append_u32_le(container, 1u);  // version
    append_u32_le(container, static_cast<std::uint32_t>(dims));
    append_u64_le(container, static_cast<std::uint64_t>(N));
    append_i64_le(container, p);
    append_i64_le(container, threshold_arg);
    append_i64_le(container, diff_arg);
    append_double_le(container, eb);
    append_u64_le(container, static_cast<std::uint64_t>(outBlockStreams.empty() ? 0 : outBlockStreams[0].size()));
    append_u64_le(container, static_cast<std::uint64_t>(outIDRel.size()));
    for (double mn : mins) {
        append_double_le(container, mn);
    }

    auto append_component = [&](const char* name, const std::string& payload) {
        container.append(name, 4);
        append_u64_le(container, static_cast<std::uint64_t>(payload.size()));
        container += payload;
    };
    append_component("BLK0", block_bytes);
    append_component("IDR0", idrel_bytes);
    append_component("REL0", rel_bytes);
    append_component("CHO0", choice_bytes);
*/
    // The first return value alone is sufficient for lcps_decompress2().
    // Keep the remaining return values as the existing component statistics.
    std::string bitstream;
    bitstream.append(ec ? "LCP2" : "LCP3", 4);
    append_u32_le(bitstream, static_cast<std::uint32_t>(dims));
    append_u64_le(bitstream, static_cast<std::uint64_t>(N));
    append_i64_le(bitstream, effective_p);
    append_double_le(bitstream, eb);
    append_i64_le(bitstream, block_split.wrap.min_value);
    append_i64_le(bitstream, block_split.wrap.max_value);
    for (double minimum : mins) {
        append_double_le(bitstream, minimum);
    }
    auto append_stream = [&](const std::string& encoded) {
        append_u64_le(bitstream, static_cast<std::uint64_t>(encoded.size()));
        bitstream += encoded;
    };
    append_stream(idrel_bytes);
    append_stream(diff_bytes);
    append_stream(keep_bytes);
    append_stream(choice2_bytes);
    append_stream(rel_bytes);
    append_u64_le(bitstream, exact_count);
    append_stream(exact_count == 0 ? std::string()
                                   : (ec ? zstd_compress_cpp(exact_corrections)
                                         : exact_corrections));

    py::list ret;
    ret.append(py::bytes(bitstream));
    ret.append(block_replacement_size);
    ret.append(static_cast<std::size_t>(idrel_bytes.size()));
    ret.append(static_cast<std::size_t>(rel_bytes.size()));
    ret.append(static_cast<std::size_t>(N - outBlockStreams[0].size()));
    return ret;
}

PYBIND11_MODULE(lcps_cpp, m) {
    m.doc() = "LCPS compression and decompression with self-contained LCP2 bitstreams.";

    m.def(
        "lcps_compress2",
        &lcps_compress2_cpp,
        py::arg("points"),
        py::arg("eb"),
        py::arg("p"),
        py::arg("threshold"),
        py::arg("diff"),
        py::arg("bl") = true,
        py::arg("l1") = true,
        py::arg("l2") = true,
        py::arg("ec") = true
    );
    m.def(
        "lcps_decompress2",
        &lcps_decompress2_cpp,
        py::arg("bitstream"),
        "Decode an LCP2 bytes object into a C-contiguous float64 (N, D) array."
    );
}
