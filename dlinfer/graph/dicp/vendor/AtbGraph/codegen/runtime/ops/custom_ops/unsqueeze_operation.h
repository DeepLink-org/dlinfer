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

}  // namespace dicp
