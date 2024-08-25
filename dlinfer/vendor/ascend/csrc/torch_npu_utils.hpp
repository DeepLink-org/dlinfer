// Copyright (c) 2024, DeepLink. All rights reserved.
#pragma once

#include <c10/core/ScalarType.h>
#include <third_party/acl/inc/acl/acl_base.h>

namespace dlinfer {
namespace ascend {

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)                            \
  _(at::ScalarType::Byte, ACL_UINT8)                                           \
  _(at::ScalarType::Char, ACL_INT8)                                            \
  _(at::ScalarType::Short, ACL_INT16)                                          \
  _(at::ScalarType::Int, ACL_INT32)                                            \
  _(at::ScalarType::Long, ACL_INT64)                                           \
  _(at::ScalarType::Half, ACL_FLOAT16)                                         \
  _(at::ScalarType::Float, ACL_FLOAT)                                          \
  _(at::ScalarType::Double, ACL_DOUBLE)                                        \
  _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)                                \
  _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)                               \
  _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)                             \
  _(at::ScalarType::Bool, ACL_BOOL)                                            \
  _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)                                   \
  _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)                                  \
  _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)                                  \
  _(at::ScalarType::BFloat16, ACL_BF16)                                        \
  _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)                                \
  _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)                                \
  _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED)                                 \
  _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED)                                 \
  _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED)                                 \
  _(at::ScalarType::Bits8, ACL_DT_UNDEFINED)                                   \
  _(at::ScalarType::Bits16, ACL_DT_UNDEFINED)                                  \
  _(at::ScalarType::Float8_e5m2, ACL_DT_UNDEFINED)                             \
  _(at::ScalarType::Float8_e4m3fn, ACL_DT_UNDEFINED)                           \
  _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)                               \
  _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable
    [static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
        AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

aclDataType convert_to_acl_data_type(const at::ScalarType &data_type);

bool is_scalar_wrapped_to_tensor(const at::Tensor &tensor);

at::Tensor copy_scalar_to_device(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type);

at::Tensor unsafe_empty_workspace(uint64_t workspace_size);

void SetDeterministic();

}  // namespace ascend
}  // namespace dlinfer
