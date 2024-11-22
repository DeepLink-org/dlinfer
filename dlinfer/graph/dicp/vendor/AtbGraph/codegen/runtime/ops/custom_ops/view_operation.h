#pragma once

#include <bits/stdint-intn.h>
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "atb/operation.h"
#include "reshape_operation.h"
#include "utils/common.h"
#include "utils/log.h"
namespace dicp {

class ViewOperation : public ReshapeOperation {
public:
    explicit ViewOperation(const std::string& name, std::vector<int64_t> viewShape);
    ~ViewOperation(){};
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;

private:
    std::vector<int64_t> shape_;
    bool needInferDim_;
    int inferDim_;
    int otherProd_ = 1;
};

inline atb::Operation* CustomViewOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::vector<int64_t> shape;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("viewShape")) {
        shape = std::move(paramJson["viewShape"].get<std::vector<int64_t>>());
    }
    DICP_LOG(INFO) << "CustomViewOperation: name: " << opName << " viewShape:" << vectorToString<int64_t>(shape);
    atb::Operation* op = new ViewOperation(opName, shape);
    return op;
}

}  // namespace dicp
