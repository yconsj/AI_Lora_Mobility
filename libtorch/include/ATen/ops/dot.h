#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>



#include <ATen/ops/dot_ops.h>

namespace at {


// aten::dot(Tensor self, Tensor tensor) -> Tensor
inline at::Tensor dot(const at::Tensor & self, const at::Tensor & tensor) {
    return at::_ops::dot::call(self, tensor);
}

// aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & dot_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & tensor) {
    return at::_ops::dot_out::call(self, tensor, out);
}
// aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & dot_outf(const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out) {
    return at::_ops::dot_out::call(self, tensor, out);
}

}