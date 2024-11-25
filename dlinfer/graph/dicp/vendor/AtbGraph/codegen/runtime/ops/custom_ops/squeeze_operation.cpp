#include "squeeze_operation.h"

#include <algorithm>
#include <iterator>

#include "ops/operation_creator.h"

namespace dicp {

SqueezeOperation::SqueezeOperation(const std::string& name, std::vector<int64_t> squeezeDim) : ReshapeOperation(name), squeezeDim_(std::move(squeezeDim)) {}

atb::Status SqueezeOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << "SqueezeOperation: " << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

    auto& oldShape = inTensorDescs.at(0).shape;
    std::vector<int64_t> dimValues(oldShape.dims, oldShape.dims + oldShape.dimNum);
    for (const auto& d : squeezeDim_) {
        int offset = d < 0 ? d + oldShape.dimNum : d;
        dimValues.erase(dimValues.begin() + offset);
    }
    outTensorDescs.at(0).shape.dimNum = dimValues.size();
    std::copy(dimValues.begin(), dimValues.end(), outTensorDescs.at(0).shape.dims);

    DICP_LOG(INFO) << "SqueezeOperation: " << opName_ << " infer shape end, out shape: " << atbDimsToString(outTensorDescs.at(0).shape);
    return atb::NO_ERROR;
}

atb::Operation* CustomSqueezeOperationCreate(const nlohmann::json& paramJson) {
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

REGISTER_OPERATION(CustomSqueezeOperation, CustomSqueezeOperationCreate);

}  // namespace dicp
