#pragma once

#include <bits/stdint-intn.h>
#include <nlohmann/json.hpp>

#include <vector>

#include "atb/operation.h"
#include "reshape_operation.h"
#include "utils/common.h"
#include "utils/log.h"
namespace dicp {

class SqueezeOperation : public ReshapeOperation {
public:
    explicit SqueezeOperation(const std::string& name, std::vector<int64_t> squeezeDim);
    ~SqueezeOperation(){};
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;

private:
    std::vector<int64_t> squeezeDim_;
};

inline atb::Operation* CustomSqueezeOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::vector<int64_t> squeezeDim;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("squeezeDim")) {
        squeezeDim = std::move(paramJson["squeezeDim"].get<std::vector<int64_t>>());
    }
    DICP_LOG(INFO) << "CustomSqueezeOperation: name: " << opName << " squeezeDim:" << vectorToString<int64_t>(squeezeDim);
    atb::Operation* op = new SqueezeOperation(opName, squeezeDim);
    return op;
}

}  // namespace dicp
