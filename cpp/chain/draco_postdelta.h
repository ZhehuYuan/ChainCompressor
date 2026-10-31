// Copyright 2016 The Draco Authors.
// Copyright 2026 OpenAI.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Header-only extraction/adaptation of the post-quantization path used by
// Draco's order-preserving point-cloud encoder:
//
//   int64 XYZ component streams in original order
//       -> wrapped delta prediction
//       -> signed-to-unsigned mapping
//       -> Draco-style Raw/Tagged selection
//       -> rANS
//
// The algorithms follow Google Draco commit
// 7d2af85eb76906b0b485a0644675e4827e4571f5, extended here to 64-bit integer
// symbols. The small DPS2 container is standalone and is NOT byte-compatible
// with a .drc stream.

#ifndef DRACO_POSTDELTA_H_
#define DRACO_POSTDELTA_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace draco_postdelta {

constexpr std::size_t kNumComponents = 3;
constexpr std::uint8_t kFormatVersion = 1;
constexpr int kMaxRawSymbolBitLength = 18;
constexpr int kDefaultCompressionLevel = 7;
constexpr std::uint32_t kMaxRansAlphabet = 1u << 18;

enum class CodingMethod : std::uint8_t {
  kRaw = 0,
  kTagged = 1,
};

[[noreturn]] inline void Fail(const std::string &message) {
  throw std::runtime_error("draco_postdelta: " + message);
}

class ByteWriter {
 public:
  void PutU8(std::uint8_t value) {
    data_.push_back(static_cast<char>(value));
  }

  void PutU64(std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      PutU8(static_cast<std::uint8_t>(value >> (8 * i)));
    }
  }

  void PutI64(std::int64_t value) {
    PutU64(static_cast<std::uint64_t>(value));
  }

  void PutVarUint(std::uint64_t value) {
    while (value >= 0x80) {
      PutU8(static_cast<std::uint8_t>(value) | 0x80u);
      value >>= 7;
    }
    PutU8(static_cast<std::uint8_t>(value));
  }

  void PutBytes(const void *src, std::size_t size) {
    if (size == 0) {
      return;
    }
    const char *bytes = static_cast<const char *>(src);
    data_.append(bytes, size);
  }

  void PutBytes(const std::vector<std::uint8_t> &bytes) {
    PutBytes(bytes.data(), bytes.size());
  }

  std::string &&Take() { return std::move(data_); }

 private:
  std::string data_;
};

inline int BitLength(std::uint64_t value) {
  if (value == 0) {
    return 1;
  }
#if defined(__GNUC__) || defined(__clang__)
  return 64 - __builtin_clzll(value);
#else
  int bits = 0;
  while (value != 0) {
    ++bits;
    value >>= 1;
  }
  return bits;
#endif
}

inline std::uint64_t SignedToSymbol(std::int64_t value) {
  if (value >= 0) {
    return static_cast<std::uint64_t>(value) << 1;
  }
  const std::uint64_t magnitude_minus_one =
      static_cast<std::uint64_t>(-(value + 1));
  return (magnitude_minus_one << 1) | 1u;
}

struct WrapParameters {
  std::int64_t min_value = 0;
  std::int64_t max_value = 0;
  std::int64_t max_difference = 1;
  std::int64_t min_correction = 0;
  std::int64_t max_correction = 0;
};

inline WrapParameters MakeWrapParameters(std::int64_t min_value,
                                         std::int64_t max_value) {
  if (min_value > max_value) {
    Fail("invalid wrap range");
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
  // Keep all wrap parameters and corrections representable in int64_t.
  if (difference >= std::numeric_limits<std::int64_t>::max()) {
    Fail("input int64 range is too wide for the wrap transform");
  }

  WrapParameters result;
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

inline std::int64_t ClampPrediction(std::int64_t value,
                                    const WrapParameters &wrap) {
  return std::min(wrap.max_value, std::max(wrap.min_value, value));
}

struct PredictedSymbols {
  WrapParameters wrap;
  std::vector<std::uint64_t> symbols;
};

inline PredictedSymbols CopyPointsWithoutPrediction(
    const std::int64_t* points,
    std::size_t num_points) {

  PredictedSymbols result;

  if (num_points == 0) {
    result.wrap = MakeWrapParameters(0, 0);
    return result;
  }

  if (points == nullptr) {
    Fail("points is null but num_points is non-zero");
  }

  if (num_points >
      std::numeric_limits<std::size_t>::max() / kNumComponents) {
    Fail("point count overflow");
  }

  const std::size_t num_values =
      num_points * kNumComponents;

  const auto minmax =
      std::minmax_element(points, points + num_values);

  result.wrap =
      MakeWrapParameters(*minmax.first, *minmax.second);

  result.symbols.resize(num_values);

  for (std::size_t i = 0; i < num_values; ++i) {
    result.symbols[i] =
        static_cast<std::uint64_t>(points[i]);
  }

  return result;
}

inline PredictedSymbols ComputePredictedSymbols(
    const std::int64_t *points, std::size_t num_points) {
  PredictedSymbols result;
  if (num_points == 0) {
    result.wrap = MakeWrapParameters(0, 0);
    return result;
  }
  if (num_points >
      std::numeric_limits<std::size_t>::max() / kNumComponents) {
    Fail("point count overflow");
  }
  const std::size_t num_values = num_points * kNumComponents;

  const auto minmax = std::minmax_element(points, points + num_values);
  result.wrap = MakeWrapParameters(*minmax.first, *minmax.second);
  result.symbols.resize(num_values);

  for (std::size_t i = 0; i < num_points; ++i) {
    for (std::size_t c = 0; c < kNumComponents; ++c) {
      const std::int64_t original = points[i * kNumComponents + c];
      const std::int64_t raw_prediction =
          i == 0 ? 0 : points[(i - 1) * kNumComponents + c];
      const std::int64_t prediction =
          ClampPrediction(raw_prediction, result.wrap);

      std::int64_t correction = original - prediction;
      if (correction < result.wrap.min_correction) {
        correction += result.wrap.max_difference;
      } else if (correction > result.wrap.max_correction) {
        correction -= result.wrap.max_difference;
      }
      result.symbols[i * kNumComponents + c] =
          SignedToSymbol(correction);
    }
  }
  return result;
}

inline std::int64_t ShannonBits(
    const std::vector<std::uint64_t> &frequencies, std::uint64_t total) {
  if (total == 0) {
    return 0;
  }
  double bits = 0.0;
  const double total_d = static_cast<double>(total);
  for (const std::uint64_t frequency : frequencies) {
    if (frequency != 0) {
      bits += static_cast<double>(frequency) *
              std::log2(static_cast<double>(frequency) / total_d);
    }
  }
  return static_cast<std::int64_t>(-bits);
}

inline std::int64_t ApproximateRansTableBits(
    std::int64_t max_value, std::int64_t num_unique_symbols) {
  return 8 * num_unique_symbols +
         8 * (num_unique_symbols +
              (max_value - num_unique_symbols) / 64);
}

struct SymbolAnalysis {
  std::vector<std::uint8_t> tags;
  std::uint64_t max_symbol = 0;
  int max_symbol_bits = 1;
  std::size_t num_unique_symbols = 0;
  std::int64_t tagged_bits = 0;
  std::int64_t raw_bits = std::numeric_limits<std::int64_t>::max();
  bool raw_supported = false;
};

inline SymbolAnalysis AnalyzeSymbols(
    const std::vector<std::uint64_t> &symbols, std::size_t num_points) {
  SymbolAnalysis analysis;
  analysis.tags.resize(num_points);
  std::array<std::uint64_t, 65> tag_frequencies{};
  std::uint64_t packed_value_bits = 0;

  for (std::size_t i = 0; i < num_points; ++i) {
    std::uint64_t max_entry = 0;
    for (std::size_t c = 0; c < kNumComponents; ++c) {
      max_entry = std::max(
          max_entry, symbols[i * kNumComponents + c]);
    }
    const int bits = BitLength(max_entry);
    analysis.tags[i] = static_cast<std::uint8_t>(bits);
    ++tag_frequencies[bits];
    packed_value_bits += static_cast<std::uint64_t>(bits) * kNumComponents;
    analysis.max_symbol = std::max(analysis.max_symbol, max_entry);
  }
  analysis.max_symbol_bits = BitLength(std::max<std::uint64_t>(
      1, analysis.max_symbol));

  std::vector<std::uint64_t> nonempty_tag_frequencies;
  for (const std::uint64_t frequency : tag_frequencies) {
    if (frequency != 0) {
      nonempty_tag_frequencies.push_back(frequency);
    }
  }
  const std::int64_t num_unique_tags =
      static_cast<std::int64_t>(nonempty_tag_frequencies.size());
  const std::int64_t tag_entropy =
      ShannonBits(nonempty_tag_frequencies, num_points);
  const std::uint64_t tagged_total =
      static_cast<std::uint64_t>(tag_entropy) +
      static_cast<std::uint64_t>(ApproximateRansTableBits(
          num_unique_tags, num_unique_tags)) +
      packed_value_bits;
  if (tagged_total >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    Fail("tagged-size estimate overflow");
  }
  analysis.tagged_bits = static_cast<std::int64_t>(tagged_total);

  // Draco's automatic selector always chooses Tagged above 18 bits. Avoiding
  // a max_symbol+1 allocation here also prevents hostile inputs from forcing
  // an enormous sparse histogram.
  if (analysis.max_symbol_bits <= kMaxRawSymbolBitLength) {
    std::vector<std::uint64_t> frequencies(
        static_cast<std::size_t>(analysis.max_symbol) + 1, 0);
    for (const std::uint64_t symbol : symbols) {
      ++frequencies[static_cast<std::size_t>(symbol)];
    }
    for (const std::uint64_t frequency : frequencies) {
      analysis.num_unique_symbols += frequency != 0;
    }
    const std::int64_t entropy = ShannonBits(frequencies, symbols.size());
    analysis.raw_bits = entropy + ApproximateRansTableBits(
                                      static_cast<std::int64_t>(
                                          analysis.max_symbol),
                                      analysis.num_unique_symbols);
    analysis.raw_supported = true;
  }
  return analysis;
}

inline int AdjustedUniqueSymbolBits(std::size_t num_unique_symbols,
                                    int compression_level) {
  if (num_unique_symbols == 0) {
    return 1;
  }
  int bits = BitLength(static_cast<std::uint64_t>(num_unique_symbols));
  if (compression_level < 4) {
    bits -= 2;
  } else if (compression_level < 6) {
    bits -= 1;
  } else if (compression_level > 9) {
    bits += 2;
  } else if (compression_level > 7) {
    bits += 1;
  }
  return std::min(kMaxRawSymbolBitLength, std::max(1, bits));
}

inline int RansPrecisionBits(int unique_symbol_bits) {
  return std::min(20, std::max(12, (3 * unique_symbol_bits) / 2));
}

struct RansTable {
  int precision_bits = 12;
  std::vector<std::uint32_t> probabilities;
  std::vector<std::uint32_t> cumulative;
};

inline RansTable NormalizeFrequencies(
    const std::vector<std::uint64_t> &input_frequencies,
    int precision_bits) {
  if (precision_bits < 12 || precision_bits > 20) {
    Fail("invalid rANS precision");
  }
  std::size_t num_symbols = input_frequencies.size();
  while (num_symbols > 0 && input_frequencies[num_symbols - 1] == 0) {
    --num_symbols;
  }
  if (num_symbols == 0) {
    Fail("cannot build an rANS table for an empty stream");
  }
  if (num_symbols > kMaxRansAlphabet) {
    Fail("rANS alphabet exceeds the standalone format limit");
  }

  std::uint64_t total_frequency = 0;
  for (std::size_t i = 0; i < num_symbols; ++i) {
    total_frequency += input_frequencies[i];
  }
  if (total_frequency == 0) {
    Fail("zero rANS total frequency");
  }

  const std::uint32_t precision = 1u << precision_bits;
  const double total_d = static_cast<double>(total_frequency);
  const double precision_d = static_cast<double>(precision);

  RansTable table;
  table.precision_bits = precision_bits;
  table.probabilities.resize(num_symbols, 0);
  std::int64_t total_probability = 0;
  for (std::size_t i = 0; i < num_symbols; ++i) {
    const std::uint64_t frequency = input_frequencies[i];
    std::uint32_t probability = static_cast<std::uint32_t>(
        static_cast<double>(frequency) / total_d * precision_d + 0.5);
    if (probability == 0 && frequency != 0) {
      probability = 1;
    }
    table.probabilities[i] = probability;
    total_probability += probability;
  }

  if (total_probability != precision) {
    std::vector<std::size_t> sorted(num_symbols);
    std::iota(sorted.begin(), sorted.end(), 0);
    std::stable_sort(sorted.begin(), sorted.end(),
                     [&](std::size_t lhs, std::size_t rhs) {
                       return table.probabilities[lhs] <
                              table.probabilities[rhs];
                     });

    if (total_probability < precision) {
      table.probabilities[sorted.back()] +=
          static_cast<std::uint32_t>(precision - total_probability);
      total_probability = precision;
    } else {
      std::int64_t error = total_probability - precision;
      while (error > 0) {
        const double relative =
            precision_d / static_cast<double>(total_probability);
        bool changed = false;
        for (std::size_t j = num_symbols; j-- > 1 && error > 0;) {
          const std::size_t symbol = sorted[j];
          std::uint32_t &probability = table.probabilities[symbol];
          if (probability <= 1) {
            if (j == num_symbols - 1) {
              Fail("rANS normalization cannot retain every used symbol");
            }
            break;
          }
          const std::uint32_t new_probability =
              static_cast<std::uint32_t>(std::floor(relative * probability));
          std::int64_t fix =
              static_cast<std::int64_t>(probability) - new_probability;
          if (fix == 0) {
            fix = 1;
          }
          if (fix >= probability) {
            fix = probability - 1;
          }
          fix = std::min(fix, error);
          probability -= static_cast<std::uint32_t>(fix);
          total_probability -= fix;
          error -= fix;
          changed = true;
        }
        if (!changed) {
          Fail("rANS normalization did not converge");
        }
      }
    }
  }

  table.cumulative.resize(num_symbols, 0);
  std::uint32_t cumulative = 0;
  for (std::size_t i = 0; i < num_symbols; ++i) {
    table.cumulative[i] = cumulative;
    cumulative += table.probabilities[i];
  }
  if (cumulative != precision) {
    Fail("normalized rANS probabilities do not sum to precision");
  }
  return table;
}

template <typename Symbol>
inline std::vector<std::uint64_t> CountSymbols(
    const std::vector<Symbol> &symbols) {
  if (symbols.empty()) {
    return {};
  }
  const std::uint64_t max_symbol = static_cast<std::uint64_t>(
      *std::max_element(symbols.begin(), symbols.end()));
  if (max_symbol >= kMaxRansAlphabet) {
    Fail("Raw rANS alphabet is too large; use Tagged coding");
  }
  std::vector<std::uint64_t> frequencies(
      static_cast<std::size_t>(max_symbol) + 1, 0);
  for (const Symbol symbol : symbols) {
    ++frequencies[static_cast<std::size_t>(symbol)];
  }
  return frequencies;
}

template <typename Symbol>
inline std::vector<std::uint8_t> RansEncode(
    const std::vector<Symbol> &symbols, const RansTable &table) {
  const std::uint32_t precision = 1u << table.precision_bits;
  const std::uint32_t lower_bound = precision * 4u;
  std::uint32_t state = lower_bound;
  std::vector<std::uint8_t> output;
  output.reserve(symbols.size() / 2 + 4);

  for (std::size_t i = symbols.size(); i-- > 0;) {
    const std::uint64_t symbol_u64 =
        static_cast<std::uint64_t>(symbols[i]);
    if (symbol_u64 >= table.probabilities.size()) {
      Fail("symbol is missing from its rANS table");
    }
    const std::size_t symbol = static_cast<std::size_t>(symbol_u64);
    if (table.probabilities[symbol] == 0) {
      Fail("symbol is missing from its rANS table");
    }
    const std::uint32_t probability = table.probabilities[symbol];
    const std::uint64_t renormalization_threshold =
        (static_cast<std::uint64_t>(lower_bound) >> table.precision_bits) *
        256u * probability;
    while (state >= renormalization_threshold) {
      output.push_back(static_cast<std::uint8_t>(state));
      state >>= 8;
    }
    const std::uint64_t next_state =
        (static_cast<std::uint64_t>(state) / probability) * precision +
        state % probability + table.cumulative[symbol];
    if (next_state >=
        static_cast<std::uint64_t>(lower_bound) * 256u) {
      Fail("rANS state overflow");
    }
    state = static_cast<std::uint32_t>(next_state);
  }

  const std::uint32_t serialized_state = state - lower_bound;
  if (serialized_state < (1u << 6)) {
    output.push_back(static_cast<std::uint8_t>(serialized_state));
  } else if (serialized_state < (1u << 14)) {
    const std::uint32_t word = (1u << 14) + serialized_state;
    output.push_back(static_cast<std::uint8_t>(word));
    output.push_back(static_cast<std::uint8_t>(word >> 8));
  } else if (serialized_state < (1u << 22)) {
    const std::uint32_t word = (2u << 22) + serialized_state;
    for (int i = 0; i < 3; ++i) {
      output.push_back(static_cast<std::uint8_t>(word >> (8 * i)));
    }
  } else if (serialized_state < (1u << 30)) {
    const std::uint32_t word = (3u << 30) + serialized_state;
    for (int i = 0; i < 4; ++i) {
      output.push_back(static_cast<std::uint8_t>(word >> (8 * i)));
    }
  } else {
    Fail("rANS final state cannot be serialized");
  }
  return output;
}

inline void EncodeProbabilityTable(
    ByteWriter &writer,
    const std::vector<std::uint32_t> &probabilities) {
  writer.PutVarUint(probabilities.size());
  for (std::size_t i = 0; i < probabilities.size(); ++i) {
    const std::uint32_t probability = probabilities[i];
    if (probability == 0) {
      std::uint32_t offset = 0;
      while (offset < 63 && i + offset + 1 < probabilities.size() &&
             probabilities[i + offset + 1] == 0) {
        ++offset;
      }
      writer.PutU8(static_cast<std::uint8_t>((offset << 2) | 3u));
      i += offset;
      continue;
    }

    int extra_bytes = 0;
    if (probability >= (1u << 6)) {
      ++extra_bytes;
      if (probability >= (1u << 14)) {
        ++extra_bytes;
      }
    }
    writer.PutU8(static_cast<std::uint8_t>(
        (probability << 2) | static_cast<std::uint32_t>(extra_bytes)));
    for (int b = 0; b < extra_bytes; ++b) {
      writer.PutU8(static_cast<std::uint8_t>(
          probability >> (8 * (b + 1) - 2)));
    }
  }
}

template <typename Symbol>
inline void EncodeRansBlock(ByteWriter &writer,
                            const std::vector<Symbol> &symbols,
                            int unique_symbol_bits) {
  const std::vector<std::uint64_t> frequencies = CountSymbols(symbols);
  const RansTable table = NormalizeFrequencies(
      frequencies, RansPrecisionBits(unique_symbol_bits));
  const std::vector<std::uint8_t> encoded = RansEncode(symbols, table);

  writer.PutU8(static_cast<std::uint8_t>(table.precision_bits));
  EncodeProbabilityTable(writer, table.probabilities);
  writer.PutVarUint(encoded.size());
  writer.PutBytes(encoded);
}

class BitWriter {
 public:
  void Put(std::uint64_t value, int bits) {
    if (bits < 1 || bits > 64) {
      Fail("invalid tagged bit width");
    }
    int remaining = bits;
    while (remaining > 0) {
      const int take = std::min(remaining, 8 - buffered_bits_);
      const std::uint64_t mask = (std::uint64_t{1} << take) - 1;
      buffer_ |= (value & mask) << buffered_bits_;
      value >>= take;
      remaining -= take;
      buffered_bits_ += take;
      if (buffered_bits_ == 8) {
        bytes_.push_back(static_cast<std::uint8_t>(buffer_));
        buffer_ = 0;
        buffered_bits_ = 0;
      }
    }
  }

  std::vector<std::uint8_t> Finish() {
    if (buffered_bits_ != 0) {
      bytes_.push_back(static_cast<std::uint8_t>(buffer_));
      buffer_ = 0;
      buffered_bits_ = 0;
    }
    return std::move(bytes_);
  }

 private:
  std::vector<std::uint8_t> bytes_;
  std::uint64_t buffer_ = 0;
  int buffered_bits_ = 0;
};

inline std::vector<std::uint8_t> PackTaggedValues(
    const std::vector<std::uint64_t> &symbols,
    const std::vector<std::uint8_t> &tags) {
  BitWriter writer;
  for (std::size_t i = 0; i < tags.size(); ++i) {
    for (std::size_t c = 0; c < kNumComponents; ++c) {
      writer.Put(symbols[i * kNumComponents + c], tags[i]);
    }
  }
  return writer.Finish();
}

inline CodingMethod ParseRequestedMethod(const std::string &method,
                                         const SymbolAnalysis &analysis) {
  if (method == "auto") {
    if (!analysis.raw_supported ||
        analysis.tagged_bits < analysis.raw_bits) {
      return CodingMethod::kTagged;
    }
    return CodingMethod::kRaw;
  }
  if (method == "raw") {
    if (!analysis.raw_supported) {
      Fail("Raw coding requires every unsigned residual to fit in 18 bits");
    }
    return CodingMethod::kRaw;
  }
  if (method == "tagged") {
    return CodingMethod::kTagged;
  }
  Fail("method must be 'auto', 'raw', or 'tagged'");
}

inline std::string CompressInterleaved(
    const std::int64_t *points, std::size_t num_points,
    const std::string &requested_method, int compression_level) {
  if (num_points != 0 && points == nullptr) {
    Fail("points is null but num_points is non-zero");
  }
  if (compression_level < 0 || compression_level > 10) {
    Fail("compression_level must be in [0, 10]");
  }
  PredictedSymbols predicted =
      ComputePredictedSymbols(points, num_points);
      //CopyPointsWithoutPrediction(points, num_points);

  CodingMethod method = CodingMethod::kRaw;
  SymbolAnalysis analysis;
  if (num_points != 0) {
    analysis = AnalyzeSymbols(predicted.symbols, num_points);
    method = ParseRequestedMethod(requested_method, analysis);
  } else {
    if (requested_method == "tagged") {
      method = CodingMethod::kTagged;
    } else if (requested_method == "auto" || requested_method == "raw") {
      method = CodingMethod::kRaw;
    } else {
      Fail("method must be 'auto', 'raw', or 'tagged'");
    }
  }

  ByteWriter writer;
  writer.PutBytes("DPS2", 4);
  writer.PutU8(kFormatVersion);
  writer.PutU8(static_cast<std::uint8_t>(method));
  writer.PutU8(static_cast<std::uint8_t>(kNumComponents));
  writer.PutU8(static_cast<std::uint8_t>(compression_level));
  writer.PutU64(num_points);
  writer.PutI64(predicted.wrap.min_value);
  writer.PutI64(predicted.wrap.max_value);

  if (num_points == 0) {
    return writer.Take();
  }

  if (method == CodingMethod::kRaw) {
    EncodeRansBlock(
        writer, predicted.symbols,
        AdjustedUniqueSymbolBits(analysis.num_unique_symbols,
                                 compression_level));
  } else {
    std::vector<std::uint32_t> tags(analysis.tags.begin(),
                                    analysis.tags.end());
    EncodeRansBlock(writer, tags, 5);
    const std::vector<std::uint8_t> values =
        PackTaggedValues(predicted.symbols, analysis.tags);
    writer.PutVarUint(values.size());
    writer.PutBytes(values);
  }
  return writer.Take();
}

inline std::string CompressXYZ(
    const std::vector<std::vector<std::int64_t>> &data,
    const std::string &method = "auto",
    int compression_level = kDefaultCompressionLevel) {
  if (data.size() != kNumComponents) {
    Fail("input must have shape (3, N): data.size() must equal 3");
  }
  const std::size_t num_points = data[0].size();
  for (std::size_t c = 1; c < kNumComponents; ++c) {
    if (data[c].size() != num_points) {
      Fail("input must have shape (3, N): all components need equal length");
    }
  }
  if (num_points >
      std::numeric_limits<std::size_t>::max() / kNumComponents) {
    Fail("point count overflow");
  }

  std::vector<std::int64_t> interleaved(num_points * kNumComponents);
  for (std::size_t i = 0; i < num_points; ++i) {
    for (std::size_t c = 0; c < kNumComponents; ++c) {
      interleaved[i * kNumComponents + c] = data[c][i];
    }
  }

  return CompressInterleaved(interleaved.data(), num_points, method,
                             compression_level);
}

// Short alias for callers that do not need the XYZ-specific name.
inline std::string Compress(
    const std::vector<std::vector<std::int64_t>> &data,
    const std::string &method = "auto",
    int compression_level = kDefaultCompressionLevel) {
  return CompressXYZ(data, method, compression_level);
}

}  // namespace draco_postdelta

#endif  // DRACO_POSTDELTA_H_
