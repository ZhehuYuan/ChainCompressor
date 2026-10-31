// Copyright 2026 OpenAI. Licensed under Apache-2.0.
// Out-of-band observation of DracoPy 2.1.0's unmodified encoding decisions.
// This state is never an attribute and is never written to an EncoderBuffer.
#ifndef DRACO_CORE_DRACO_IDS_TRACE_H_
#define DRACO_CORE_DRACO_IDS_TRACE_H_

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace draco_ids_trace {
struct State {
  std::vector<std::int32_t> original_to_unique;
  std::vector<std::int32_t> slot_to_unique;
  std::vector<std::int32_t> decoded_to_unique;
  std::uintptr_t base = 0;
  std::size_t stride = 0;
  bool kd_tree_seen = false;
  bool invalid_swap = false;
};
inline thread_local State* active = nullptr;

class Scope {
 public:
  explicit Scope(State& value) : previous_(active) { active = &value; }
  ~Scope() { active = previous_; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
 private:
  State* previous_;
};

template<class Map>
inline void CaptureDeduplication(const Map& map, std::size_t n) {
  if (!active) return;
  active->original_to_unique.resize(n);
  for (std::size_t i = 0; i < n; ++i)
    active->original_to_unique[i] = static_cast<std::int32_t>(map.data()[i].value());
}

inline void BeginPoints(const void* base, std::size_t n, std::size_t dimensions) {
  if (!active) return;
  active->base = reinterpret_cast<std::uintptr_t>(base);
  active->stride = dimensions * sizeof(std::uint32_t);
  active->slot_to_unique.resize(n);
  std::iota(active->slot_to_unique.begin(), active->slot_to_unique.end(), std::int32_t{0});
  active->decoded_to_unique.clear();
  active->decoded_to_unique.reserve(n);
  active->kd_tree_seen = true;
}

inline void ObserveSwap(const void* a, const void* b) noexcept {
  if (!active || !active->kd_tree_seen) return;
  const auto x = reinterpret_cast<std::uintptr_t>(a) - active->base;
  const auto y = reinterpret_cast<std::uintptr_t>(b) - active->base;
  const auto stride = active->stride;
  if (!stride || x % stride || y % stride ||
      x / stride >= active->slot_to_unique.size() ||
      y / stride >= active->slot_to_unique.size()) {
    active->invalid_swap = true;
    return;
  }
  std::swap(active->slot_to_unique[x / stride], active->slot_to_unique[y / stride]);
}

inline void EmitLeaf(const void* begin, std::size_t count) {
  if (!active || !active->kd_tree_seen) return;
  const auto offset = reinterpret_cast<std::uintptr_t>(begin) - active->base;
  const auto stride = active->stride;
  if (!stride || offset % stride || offset / stride > active->slot_to_unique.size() ||
      count > active->slot_to_unique.size() - offset / stride)
    throw std::runtime_error("draco_ids: invalid traced leaf");
  const auto first = active->slot_to_unique.begin() + offset / stride;
  active->decoded_to_unique.insert(active->decoded_to_unique.end(), first, first + count);
}
}  // namespace draco_ids_trace
#endif
