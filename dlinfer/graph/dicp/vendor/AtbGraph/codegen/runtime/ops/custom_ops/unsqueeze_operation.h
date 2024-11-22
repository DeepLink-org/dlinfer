#pragma once

#include <bits/stdint-intn.h>
#include <nlohmann/json.hpp>

#include <vector>

#include "atb/operation.h"
#include "reshape_operation.h"
#include "utils/common.h"
#include "utils/log.h"
namespace dicp {

class UnsqueezeOperation : public ReshapeOperation {
public:
    explicit UnsqueezeOperation(const std::string& name, std::vector<int64_t> unsqueezeDim);
    ~UnsqueezeOperation(){};
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;

private:
    std::vector<int64_t> unsqueezeDim_;
};

inline atb::Operation* CustomUnsqueezeOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::vector<int64_t> unsqueezeDim;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("unsqueezeDim")) {
        unsqueezeDim = std::move(paramJson["unsqueezeDim"].get<std::vector<int64_t>>());
    }
    DICP_LOG(INFO) << "CustomUnsqueezeOperation: name: " << opName << " unsqueezeDim:" << vectorToString<int64_t>(unsqueezeDim);
    atb::Operation* op = new UnsqueezeOperation(opName, unsqueezeDim);
    return op;
}

}  // namespace dicp
