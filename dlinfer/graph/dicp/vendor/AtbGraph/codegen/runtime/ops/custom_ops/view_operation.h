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

}  // namespace dicp
