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



#include <ATen/ops/zeros_ops.h>

namespace at {


// aten::zeros.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor zeros(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options={}) {
    return at::_ops::zeros_names::call(size, names, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
// aten::zeros.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor zeros(at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::zeros_names::call(size, names, dtype, layout, device, pin_memory);
}

// aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor zeros(at::IntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::zeros::call(c10::fromIntArrayRefSlow(size), c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor zeros(at::IntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::zeros::call(c10::fromIntArrayRefSlow(size), c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor zeros(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::zeros::call(c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor zeros(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::zeros::call(c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
  }
}

// aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor zeros_symint(c10::SymIntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::zeros::call(size, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor zeros(c10::SymIntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::zeros::call(size, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor zeros_symint(c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::zeros::call(size, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor zeros(c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::zeros::call(size, dtype, layout, device, pin_memory);
  }
}

// aten::zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & zeros_out(at::Tensor & out, at::IntArrayRef size) {
    return at::_ops::zeros_out::call(c10::fromIntArrayRefSlow(size), out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & zeros_out(at::Tensor & out, at::IntArrayRef size) {
    return at::_ops::zeros_out::call(c10::fromIntArrayRefSlow(size), out);
  }
}

// aten::zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & zeros_outf(at::IntArrayRef size, at::Tensor & out) {
    return at::_ops::zeros_out::call(c10::fromIntArrayRefSlow(size), out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & zeros_outf(at::IntArrayRef size, at::Tensor & out) {
    return at::_ops::zeros_out::call(c10::fromIntArrayRefSlow(size), out);
  }
}

// aten::zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & zeros_symint_out(at::Tensor & out, c10::SymIntArrayRef size) {
    return at::_ops::zeros_out::call(size, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & zeros_out(at::Tensor & out, c10::SymIntArrayRef size) {
    return at::_ops::zeros_out::call(size, out);
  }
}

// aten::zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & zeros_symint_outf(c10::SymIntArrayRef size, at::Tensor & out) {
    return at::_ops::zeros_out::call(size, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & zeros_outf(c10::SymIntArrayRef size, at::Tensor & out) {
    return at::_ops::zeros_out::call(size, out);
  }
}

// aten::zeros.names_out(int[] size, *, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & zeros_out(at::Tensor & out, at::IntArrayRef size, ::std::optional<at::DimnameList> names) {
    return at::_ops::zeros_names_out::call(size, names, out);
}
// aten::zeros.names_out(int[] size, *, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & zeros_outf(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
    return at::_ops::zeros_names_out::call(size, names, out);
}

}