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



#include <ATen/ops/concatenate_ops.h>

namespace at {


// aten::concatenate(Tensor[] tensors, int dim=0) -> Tensor
inline at::Tensor concatenate(at::TensorList tensors, int64_t dim=0) {
    return at::_ops::concatenate::call(tensors, dim);
}

// aten::concatenate.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & concatenate_out(at::Tensor & out, at::TensorList tensors, int64_t dim=0) {
    return at::_ops::concatenate_out::call(tensors, dim, out);
}
// aten::concatenate.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & concatenate_outf(at::TensorList tensors, int64_t dim, at::Tensor & out) {
    return at::_ops::concatenate_out::call(tensors, dim, out);
}

// aten::concatenate.names(Tensor[] tensors, Dimname dim) -> Tensor
inline at::Tensor concatenate(at::TensorList tensors, at::Dimname dim) {
    return at::_ops::concatenate_names::call(tensors, dim);
}

// aten::concatenate.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & concatenate_out(at::Tensor & out, at::TensorList tensors, at::Dimname dim) {
    return at::_ops::concatenate_names_out::call(tensors, dim, out);
}
// aten::concatenate.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & concatenate_outf(at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
    return at::_ops::concatenate_names_out::call(tensors, dim, out);
}

}