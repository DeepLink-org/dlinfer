#include "unsqueeze_operation.h"

#include <algorithm>
#include <iterator>

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

}  // namespace dicp
