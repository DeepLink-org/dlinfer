#include "slice_scatter_operation.h"

#include <cstddef>

#include "aclnnop/aclnn_strided_slice_assign_v2.h"
#include "ops/operation_creator.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "utils/common.h"
#include "utils/log.h"

namespace dicp {

const int NUM0 = 0;
const int NUM1 = 1;
const int NUM2 = 2;

SliceScatterOperation::SliceScatterOperation(const std::string& name, int64_t dim, int64_t start, int64_t end, int64_t step)
    : AclNnOperation(name), opName_(name), dim_(dim), start_(start), end_(end), step_(step) {}

SliceScatterOperation::~SliceScatterOperation() {
    if (beginArray_ != nullptr) {
        aclDestroyIntArray(beginArray_);
    }
    if (endArray_ != nullptr) {
        aclDestroyIntArray(endArray_);
    }
    if (stridesArray_ != nullptr) {
        aclDestroyIntArray(stridesArray_);
    }
    if (axesArray_ != nullptr) {
        aclDestroyIntArray(axesArray_);
    }
}

std::string SliceScatterOperation::GetName() const { return opName_; }

atb::Status SliceScatterOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    auto rank = inTensorDescs.at(0).shape.dimNum;
    beginVec_.resize(rank), endVec_.resize(rank);
    stridesVec_.resize(rank), axesVec_.resize(rank);
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
        beginVec_.at(i) = i == dim_ ? start_ : NUM0;
        endVec_.at(i) = i == dim_ ? end_ : outTensorDescs.at(0).shape.dims[i];
        stridesVec_.at(i) = NUM1;
        axesVec_.at(i) = i;
    }
    DICP_LOG(INFO) << "SliceScatterOperationCreate: name: " << opName_ << ", begin:" << vectorToString(beginVec_) << ", end:" << vectorToString(endVec_)
                   << ", strides:" << vectorToString(stridesVec_) << ", axes:" << vectorToString(axesVec_);
    beginArray_ = aclCreateIntArray(beginVec_.data(), beginVec_.size());
    endArray_ = aclCreateIntArray(endVec_.data(), endVec_.size());
    stridesArray_ = aclCreateIntArray(stridesVec_.data(), stridesVec_.size());
    axesArray_ = aclCreateIntArray(axesVec_.data(), axesVec_.size());
    return 0;
}

uint32_t SliceScatterOperation::GetInputNum() const { return NUM2; }

uint32_t SliceScatterOperation::GetOutputNum() const { return NUM1; }

int SliceScatterOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnStridedSliceAssignV2GetWorkspaceSize start";
    int ret = aclnnStridedSliceAssignV2GetWorkspaceSize(
        aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, beginArray_, endArray_, stridesArray_, axesArray_, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnStridedSliceAssignV2GetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int SliceScatterOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnStridedSliceAssignV2 start";
    int ret = aclnnStridedSliceAssignV2(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnStridedSliceAssignV2 end, ret:" << ret;
    return ret;
}

atb::Operation* SliceScatterOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t dim, start, end, step;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("dim")) {
        dim = paramJson["dim"].get<std::int64_t>();
    }
    if (paramJson.contains("start")) {
        start = paramJson["start"].get<std::int64_t>();
    }
    if (paramJson.contains("end")) {
        end = paramJson["end"].get<std::int64_t>();
    }
    if (paramJson.contains("step")) {
        step = paramJson["step"].get<std::int64_t>();
    }
    DICP_LOG(INFO) << "SliceScatterOperationCreate: name: " << opName << ", dim:" << dim << ", start:" << start << ", end:" << end << ", step:" << step;
    atb::Operation* op = new SliceScatterOperation(opName, dim, start, end, step);
    return op;
}

REGISTER_OPERATION(SliceScatterOperation, SliceScatterOperationCreate);

}  // namespace dicp
