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
TORCH_API void _backward(const at::Tensor & self, at::TensorList inputs, const ::std::optional<at::Tensor> & gradient={}, ::std::optional<bool> retain_graph=::std::nullopt, bool create_graph=false);
} // namespace native
} // namespace at