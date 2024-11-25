#include "unsqueeze_operation.h"

#include <algorithm>
#include <iterator>

#include "ops/operation_creator.h"

namespace dicp {

UnsqueezeOperation::UnsqueezeOperation(const std::string& name, std::vector<int64_t> unsqueezeDim)
    : ReshapeOperation(name), unsqueezeDim_(std::move(unsqueezeDim)) {}

atb::Status UnsqueezeOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << "UnsqueezeOperation: " << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

    auto& oldShape = inTensorDescs.at(0).shape;
    std::vector<int64_t> dimValues(oldShape.dims, oldShape.dims + oldShape.dimNum);
    for (const auto& d : unsqueezeDim_) {
        int offset = d < 0 ? d + oldShape.dimNum + 1 : d;
        dimValues.insert(dimValues.begin() + offset, 1);
    }
    outTensorDescs.at(0).shape.dimNum = dimValues.size();
    std::copy(dimValues.begin(), dimValues.end(), outTensorDescs.at(0).shape.dims);

    DICP_LOG(INFO) << "UnsqueezeOperation: " << opName_ << " infer shape end, out shape: " << atbDimsToString(outTensorDescs.at(0).shape);
    return atb::NO_ERROR;
}

atb::Operation* CustomUnsqueezeOperationCreate(const nlohmann::json& paramJson) {
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

REGISTER_OPERATION(CustomUnsqueezeOperation, CustomUnsqueezeOperationCreate);

}  // namespace dicp
