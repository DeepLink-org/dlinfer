#include "squeeze_operation.h"

#include <algorithm>
#include <iterator>

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

}  // namespace dicp
