#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor bartlett_window(int64_t window_length, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={});
TORCH_API at::Tensor & bartlett_window_out(int64_t window_length, at::Tensor & out);
TORCH_API at::Tensor bartlett_window(int64_t window_length, bool periodic, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={});
TORCH_API at::Tensor & bartlett_window_periodic_out(int64_t window_length, bool periodic, at::Tensor & out);
} // namespace native
} // namespace at