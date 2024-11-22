#include "view_operation.h"

#include <algorithm>
#include <iterator>

#include "reshape_operation.h"

namespace dicp {

ViewOperation::ViewOperation(const std::string& name, std::vector<int64_t> viewShape) : ReshapeOperation(name), shape_(std::move(viewShape)) {
    auto it = std::find(shape_.begin(), shape_.end(), -1);
    needInferDim_ = (it != shape_.end());
    inferDim_ = needInferDim_ ? std::distance(shape_.begin(), it) : -1;

    if (needInferDim_) {
        otherProd_ = 1;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i != inferDim_) {
                otherProd_ *= shape_[i];
            }
        }
    }
}

atb::Status ViewOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << "ViewOperation: " << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = shape_.size();

    if (needInferDim_) {
        auto& oldShape = inTensorDescs.at(0).shape;
        int64_t totalValue = 1;
        for (size_t i = 0; i < oldShape.dimNum; ++i) {
            totalValue *= oldShape.dims[i];
        }
        outTensorDescs.at(0).shape.dims[inferDim_] = totalValue / otherProd_;
    }
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (shape_[i] != -1) {
            outTensorDescs.at(0).shape.dims[i] = shape_[i];
        }
    }
    DICP_LOG(INFO) << "ViewOperation: " << opName_ << " infer shape end, out shape: " << atbDimsToString(outTensorDescs.at(0).shape);
    return atb::NO_ERROR;
}

}  // namespace dicp
