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
TORCH_API at::Tensor & _nested_from_padded_and_nested_example_out(const at::Tensor & padded, const at::Tensor & nt_example, at::Tensor & out);
TORCH_API at::Tensor NestedTensor_from_padded_and_nested_example(const at::Tensor & padded, const at::Tensor & nt_example);
} // namespace native
} // namespace at