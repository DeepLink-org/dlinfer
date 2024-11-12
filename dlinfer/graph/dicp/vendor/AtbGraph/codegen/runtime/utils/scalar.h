#pragma once

#include <aclnn/acl_meta.h>
#include <aclnn/opdev/bfloat16.h>
#include <half.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace dicp {

class DICPScalar {
public:
    union ScalarValue {
        int64_t int64Value;
        int32_t int32Value;
        float floatValue;
        half_float::half halfValue;
        op::bfloat16 bf16Value;

        ScalarValue();
    };

    enum class ValueType { INT64, INT32, FLOAT, FLOAT16, BF16, UNKNOWN };

    DICPScalar();

    template <typename T>
    explicit DICPScalar(T val);

    DICPScalar(float value, std::string_view dtype);

    bool isInt64() const;
    bool isInt32() const;
    bool isFloat() const;
    bool isFloat16() const;
    bool isBF16() const;

    aclDataType getDataType() const;
    void* getValuePtr();
    const void* getValuePtr() const;
    std::string getTypeString() const;

private:
    ScalarValue value;
    aclDataType dataType;
    ValueType currentType;

    static ValueType parseValueType(std::string_view dtype);
    static aclDataType getAclDataTypeFromValueType(ValueType vtype);

    template <typename T>
    static constexpr bool is_supported_type() {
        return std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, float> || std::is_same_v<T, half_float::half> ||
               std::is_same_v<T, op::bfloat16>;
    }

    template <typename T>
    static ValueType getValueType();

    template <typename T>
    static aclDataType getAclDataType();
};

template <typename T>
DICPScalar::DICPScalar(T val) {
    static_assert(is_supported_type<T>(), "Unsupported type for DICPScalar");

    currentType = getValueType<T>();
    dataType = getAclDataType<T>();

    if constexpr (std::is_same_v<T, int64_t>) {
        value.int64Value = val;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        value.int32Value = val;
    } else if constexpr (std::is_same_v<T, float>) {
        value.floatValue = val;
    } else if constexpr (std::is_same_v<T, half_float::half>) {
        value.halfValue = val;
    } else if constexpr (std::is_same_v<T, op::bfloat16>) {
        value.bf16Value = val;
    }
}

template <typename T>
DICPScalar::ValueType DICPScalar::getValueType() {
    if constexpr (std::is_same_v<T, int64_t>)
        return ValueType::INT64;
    else if constexpr (std::is_same_v<T, int32_t>)
        return ValueType::INT32;
    else if constexpr (std::is_same_v<T, float>)
        return ValueType::FLOAT;
    else if constexpr (std::is_same_v<T, half_float::half>)
        return ValueType::FLOAT16;
    else if constexpr (std::is_same_v<T, op::bfloat16>)
        return ValueType::BF16;
    else
        return ValueType::UNKNOWN;
}

template <typename T>
aclDataType DICPScalar::getAclDataType() {
    if constexpr (std::is_same_v<T, int64_t>)
        return aclDataType::ACL_INT64;
    else if constexpr (std::is_same_v<T, int32_t>)
        return aclDataType::ACL_INT32;
    else if constexpr (std::is_same_v<T, float>)
        return aclDataType::ACL_FLOAT;
    else if constexpr (std::is_same_v<T, half_float::half>)
        return aclDataType::ACL_FLOAT16;
    else if constexpr (std::is_same_v<T, op::bfloat16>)
        return aclDataType::ACL_BF16;
    else
        throw std::invalid_argument("Unsupported type");
}

}  // namespace dicp
