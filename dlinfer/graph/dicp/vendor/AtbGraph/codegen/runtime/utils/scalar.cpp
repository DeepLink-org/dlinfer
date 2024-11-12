#include "utils/scalar.h"

#include <algorithm>
#include <cctype>

namespace dicp {

DICPScalar::ScalarValue::ScalarValue() : int64Value(0) {}

DICPScalar::DICPScalar() : currentType(ValueType::UNKNOWN), dataType(ACL_INT64) {}

DICPScalar::ValueType DICPScalar::parseValueType(std::string_view dtype) {
    std::string dtype_upper(dtype);
    std::transform(dtype_upper.begin(), dtype_upper.end(), dtype_upper.begin(), [](unsigned char c) { return std::toupper(c); });

    if (dtype_upper == "INT64") return ValueType::INT64;
    if (dtype_upper == "INT32") return ValueType::INT32;
    if (dtype_upper == "FLOAT") return ValueType::FLOAT;
    if (dtype_upper == "FLOAT16") return ValueType::FLOAT16;
    if (dtype_upper == "BF16") return ValueType::BF16;

    throw std::invalid_argument("Unsupported dtype: " + std::string(dtype));
}

aclDataType DICPScalar::getAclDataTypeFromValueType(ValueType vtype) {
    switch (vtype) {
        case ValueType::INT64:
            return ACL_INT64;
        case ValueType::INT32:
            return ACL_INT32;
        case ValueType::FLOAT:
            return ACL_FLOAT;
        case ValueType::FLOAT16:
            return ACL_FLOAT16;
        case ValueType::BF16:
            return ACL_BF16;
        default:
            throw std::invalid_argument("Invalid ValueType");
    }
}

DICPScalar::DICPScalar(float value, std::string_view dtype) {
    currentType = parseValueType(dtype);
    dataType = getAclDataTypeFromValueType(currentType);

    switch (currentType) {
        case ValueType::INT64:
            this->value.int64Value = static_cast<int64_t>(value);
            break;
        case ValueType::INT32:
            this->value.int32Value = static_cast<int32_t>(value);
            break;
        case ValueType::FLOAT:
            this->value.floatValue = value;
            break;
        case ValueType::FLOAT16:
            this->value.halfValue = half_float::half(value);
            break;
        case ValueType::BF16:
            this->value.halfValue = op::bfloat16(value);
            break;
        default:
            throw std::invalid_argument("Invalid target type");
    }
}

bool DICPScalar::isInt64() const { return currentType == ValueType::INT64; }

bool DICPScalar::isInt32() const { return currentType == ValueType::INT32; }

bool DICPScalar::isFloat() const { return currentType == ValueType::FLOAT; }

bool DICPScalar::isFloat16() const { return currentType == ValueType::FLOAT16; }

bool DICPScalar::isBF16() const { return currentType == ValueType::BF16; }

aclDataType DICPScalar::getDataType() const { return dataType; }

void* DICPScalar::getValuePtr() { return &value; }

const void* DICPScalar::getValuePtr() const { return &value; }

std::string DICPScalar::getTypeString() const {
    switch (currentType) {
        case ValueType::INT64:
            return "INT64";
        case ValueType::INT32:
            return "INT32";
        case ValueType::FLOAT:
            return "FLOAT";
        case ValueType::FLOAT16:
            return "FLOAT16";
        case ValueType::BF16:
            return "BF16";
        default:
            return "UNKNOWN";
    }
}

}  // namespace dicp
