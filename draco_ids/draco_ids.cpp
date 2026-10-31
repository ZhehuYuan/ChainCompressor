// Copyright 2026 OpenAI. Licensed under Apache-2.0.
// Point-cloud wrapper derived from seung-lab/DracoPy 2.1.0, src/DracoPy.h.
// Google Draco sources and licenses are in vendor/draco.
// Build: make PYTHON=python3  (see README.md).
// Public API: float32 NumPy (N,3), float scalar settings, int32 IDs/settings.
// Internal Draco wire-format types are preserved for byte compatibility.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "draco/compression/decode.h"
#include "draco/compression/encode.h"
#include "draco/core/draco_ids_trace.h"
#include "draco/metadata/geometry_metadata.h"
#include "draco/point_cloud/point_cloud_builder.h"

namespace py = pybind11;
namespace {
using Coordinates = py::array_t<float, 0>;
using Int = std::int32_t;
// The pinned Draco implementation has signed int counters for N*3.
constexpr std::size_t kMaxPoints = std::numeric_limits<Int>::max() / 3;
static_assert(sizeof(float) == 4 && sizeof(Int) == 4, "32-bit public API required");

std::vector<float> copy_coordinates(const Coordinates& data) {
  const auto view = data.request();
  if (view.ndim != 2 || view.shape[1] != 3)
    throw py::value_error("data must be a NumPy float32 array of shape (N, 3)");
  if (view.shape[0] <= 0)
    throw py::value_error("data must contain at least one point (as required by DracoPy)");
  if (static_cast<std::size_t>(view.shape[0]) > kMaxPoints)
    throw std::overflow_error("N exceeds the pinned Draco int32 N*3 limit (715827882)");
  std::vector<float> values(static_cast<std::size_t>(view.shape[0]) * 3);
  const auto* base = static_cast<const char*>(view.ptr);
  for (py::ssize_t i = 0; i < view.shape[0]; ++i) {
    for (Int d = 0; d < 3; ++d) {
      float value;
      std::memcpy(&value, base + i * view.strides[0] + d * view.strides[1], sizeof(float));
      if (!std::isfinite(value)) throw py::value_error("coordinates must be finite");
      values[3 * static_cast<std::size_t>(i) + d] = value;
    }
  }
  return values;
}

std::array<float, 3> get_origin(const Coordinates& data, const py::object& origin) {
  // DracoPy computes np.min(points, axis=0) before its C++ encode call.
  // Use the same reduction, including NumPy's treatment of signed zero.
  py::object array = origin.is_none()
      ? py::module_::import("numpy").attr("min")(data, py::arg("axis") = 0)
      : origin;
  if (!py::isinstance<Coordinates>(array))
    throw py::type_error("quantization_origin must be None or a NumPy float32 array");
  const auto view = py::cast<Coordinates>(array).request();
  if (view.ndim != 1 || view.shape[0] != 3)
    throw py::value_error("quantization_origin must have shape (3,)");
  std::array<float, 3> result;
  for (Int d = 0; d < 3; ++d) {
    std::memcpy(&result[d], static_cast<const char*>(view.ptr) + d * view.strides[0], sizeof(float));
    if (!std::isfinite(result[d])) throw py::value_error("quantization_origin must be finite");
  }
  return result;
}

void setup_encoder(draco::PointCloud& cloud, draco::Encoder& encoder,
                   Int bits, float range, const std::array<float, 3>& origin,
                   bool create_metadata) {
  encoder.SetSpeedOptions(0, 0);  // DracoPy speed = 10 - compression_level.
  if (create_metadata)
    cloud.AddMetadata(std::unique_ptr<draco::GeometryMetadata>(new draco::GeometryMetadata()));
  if (range <= 0.0f) {
    encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, bits);
  } else {
    encoder.SetAttributeExplicitQuantization(draco::GeometryAttribute::POSITION,
                                             bits, 3, origin.data(), range);
    if (create_metadata) {
      // These fields are doubles in DracoPy's *serialized metadata*. Keeping
      // that representation is necessary for identical compressed bytes.
      cloud.metadata()->AddEntryDouble("quantization_range", range);
      const std::vector<double> metadata_origin(origin.begin(), origin.end());
      cloud.metadata()->AddEntryDoubleArray("quantization_origin", metadata_origin);
    }
  }
  if (create_metadata) cloud.metadata()->AddEntryInt("quantization_bits", bits);
}

py::tuple encode(const Coordinates& data, Int quantization_bits,
                 float quantization_range, const py::object& quantization_origin,
                 bool create_metadata) {
  if (quantization_bits < 0 || quantization_bits > 30)
    throw py::value_error("quantization_bits must be in [0, 30]");
  if (!std::isfinite(quantization_range))
    throw py::value_error("quantization_range must be finite");
  const auto input = copy_coordinates(data);
  const auto origin = get_origin(data, quantization_origin);
  const auto n = input.size() / 3;
  std::string compressed;
  std::vector<Int> ids;
  {
    py::gil_scoped_release release;
    draco_ids_trace::State trace;
    draco_ids_trace::Scope scope(trace);
    draco::PointCloudBuilder builder;
    builder.Start(static_cast<std::uint32_t>(n));
    const Int attr_id = builder.AddAttribute(draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32);
    if (attr_id < 0) throw std::runtime_error("failed to create Draco position attribute");
    builder.SetAttributeUniqueId(attr_id, 0);
    for (draco::PointIndex i(0); i.value() < n; ++i)
      builder.SetAttributeValueForPoint(attr_id, i, input.data() + 3 * static_cast<std::size_t>(i.value()));
    // Exactly matches DracoPy's Finalize(!preserve_order), including deduplication.
    auto cloud = builder.Finalize(true);
    if (!cloud) throw std::runtime_error("failed to build Draco point cloud");
    draco::Encoder encoder;
    setup_encoder(*cloud, encoder, quantization_bits, quantization_range, origin, create_metadata);
    draco::EncoderBuffer buffer;
    const auto status = encoder.EncodePointCloudToBuffer(*cloud, &buffer);
    if (!status.ok()) throw std::runtime_error("Draco encoding failed: " + status.error_msg_string());
    compressed.assign(buffer.data(), buffer.size());

    const auto unique_count = static_cast<std::size_t>(cloud->num_points());
    if (trace.original_to_unique.size() != n || trace.invalid_swap)
      throw std::runtime_error("draco_ids: missing or invalid deduplication trace");
    if (!trace.kd_tree_seen) {
      // With quantization_bits=0, DracoPy selects the sequential float codec.
      if (quantization_bits != 0)
        throw std::runtime_error("draco_ids: unexpected encoder without a kd-tree trace");
      trace.decoded_to_unique.resize(unique_count);
      std::iota(trace.decoded_to_unique.begin(), trace.decoded_to_unique.end(), Int{0});
    }
    if (trace.decoded_to_unique.size() != unique_count)
      throw std::runtime_error("draco_ids: incomplete point-order trace");
    std::vector<Int> unique_to_decoded(unique_count, -1);
    for (std::size_t j = 0; j < unique_count; ++j) {
      const auto u = trace.decoded_to_unique[j];
      if (u < 0 || static_cast<std::size_t>(u) >= unique_count || unique_to_decoded[u] != -1)
        throw std::runtime_error("draco_ids: invalid point-order permutation");
      unique_to_decoded[u] = static_cast<Int>(j);
    }
    ids.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      const auto u = trace.original_to_unique[i];
      if (u < 0 || static_cast<std::size_t>(u) >= unique_count)
        throw std::runtime_error("draco_ids: invalid original point mapping");
      ids[i] = unique_to_decoded[u];
    }
  }
  py::array_t<Int> result_ids(static_cast<py::ssize_t>(n));
  std::memcpy(result_ids.mutable_data(), ids.data(), ids.size() * sizeof(Int));
  return py::make_tuple(py::bytes(compressed), std::move(result_ids));
}

py::array_t<float> decode(const py::bytes& compressed) {
  const std::string bytes = compressed;
  std::vector<float> points;
  std::size_t n;
  {
    py::gil_scoped_release release;
    draco::DecoderBuffer buffer;
    buffer.Init(bytes.data(), bytes.size());
    const auto type = draco::Decoder::GetEncodedGeometryType(&buffer);
    if (!type.ok() || type.value() != draco::POINT_CLOUD)
      throw std::runtime_error("decode expects a Draco point-cloud stream");
    draco::Decoder decoder;
    auto decoded = decoder.DecodePointCloudFromBuffer(&buffer);
    if (!decoded.ok()) throw std::runtime_error("Draco decoding failed: " + decoded.status().error_msg_string());
    auto cloud = std::move(decoded).value();
    n = cloud->num_points();
    if (n > kMaxPoints) throw std::overflow_error("decoded point count exceeds int32 N*3 limit");
    const auto* attr = cloud->GetNamedAttribute(draco::GeometryAttribute::POSITION);
    if (!attr || attr->num_components() != 3 || attr->data_type() != draco::DT_FLOAT32)
      throw std::runtime_error("decode expects three-component float32 positions");
    points.resize(3 * n);
    for (draco::PointIndex i(0); i.value() < n; ++i) {
      if (!attr->ConvertValue<float>(attr->mapped_index(i), 3, points.data() + 3 * static_cast<std::size_t>(i.value())))
        throw std::runtime_error("failed to decode a position");
    }
  }
  py::array_t<float> result({static_cast<py::ssize_t>(n), py::ssize_t{3}});
  if (!points.empty()) std::memcpy(result.mutable_data(), points.data(), points.size() * sizeof(float));
  return result;
}
}  // namespace

PYBIND11_MODULE(draco_ids, m) {
  m.doc() = "DracoPy-compatible point-cloud encoding, level=10, preserve_order=False, with original-to-decoded int32 IDs.";
  m.attr("reference_dracopy_version") = "2.1.0";
  m.def("encode", &encode,
        py::arg("data").noconvert(), py::arg("quantization_bits") = Int{14},
        py::arg("quantization_range") = -1.0f, py::arg("quantization_origin") = py::none(),
        py::arg("create_metadata") = false,
        R"doc(Return (compressed_bytes, ids) for float32 NumPy coordinates (N,3).

compression_level=10 and preserve_order=False are fixed. Quantization
parameters match DracoPy 2.1.0's point-cloud encode API. Public numerical
arrays are native-endian float32/int32; scalar settings use C++ float/int32.
quantization_origin, when supplied, must be a float32 NumPy array of shape (3,).
No faces, colors or other attributes are encoded. Inputs must be finite.

ids is an owning int32 NumPy array of shape (N,): ids[i] is the decoded
row for original particle i. Restore with decoded[ids]. This also expands
duplicate input particles merged by DracoPy. IDs are separate from bytes;
they are not a Draco attribute. This is the inverse of the earlier LCP-S
permutation convention. For unique inputs, ids is a permutation.
)doc");
  m.def("decode", &decode, py::arg("compressed").noconvert(),
        "Decode to an owning NumPy float32 (M,3) array, in native Draco decode order.");
}
