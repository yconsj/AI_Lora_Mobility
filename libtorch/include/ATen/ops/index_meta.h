#pragma once

// @generated by torchgen/gen.py from NativeMetaFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <tuple>
#include <vector>

namespace at {
namespace meta {

struct TORCH_API structured_index_Tensor : public TensorIteratorBase {
    
                template <bool SIZES = false, bool STRIDES = false>
                struct TORCH_API precompute_out {
                    
                    precompute_out<true, STRIDES> set_sizes(at::DimVector value) {
                        static_assert(SIZES == false, "sizes already set");
                        precompute_out<true, STRIDES> ret;
ret.sizes = value;
ret.strides = this->strides;
return ret;
                    }
                

                    precompute_out<SIZES, true> set_strides(at::DimVector value) {
                        static_assert(STRIDES == false, "strides already set");
                        precompute_out<SIZES, true> ret;
ret.sizes = this->sizes;
ret.strides = value;
return ret;
                    }
                
                    at::DimVector sizes;
at::DimVector strides;
            };
    using meta_return_ty = precompute_out <true, true>;
    meta_return_ty meta(const at::Tensor & self, at::IOptTensorListRef indices);
};

} // namespace native
} // namespace at