#include "slice_scatter_operation.h"

#include <cstddef>

#include "aclnnop/aclnn_strided_slice_assign_v2.h"
#include "ops/operation_creator.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "utils/common.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

SliceScatterOperation::SliceScatterOperation(const std::string& name, std::vector<int64_t> beginVec, std::vector<int64_t> endVec,
                                             std::vector<int64_t> stridesVec, std::vector<int64_t> axesVec)
    : AclNnOperation(name), opName_(name), beginVec_(beginVec), endVec_(endVec), stridesVec_(stridesVec), axesVec_(axesVec) {}

SliceScatterOperation::~SliceScatterOperation() {}

std::string SliceScatterOperation::GetName() const { return opName_; }

atb::Status SliceScatterOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
        endVec_.at(i) = endVec_[i] == -1 ? inTensorDescs.at(0).shape.dims[i] : endVec_[i];
    }
    DICP_LOG(INFO) << "SliceScatterOperationCreate: name: " << opName_ << ", begin:" << vectorToString(beginVec_) << ", end:" << vectorToString(endVec_)
                   << ", strides:" << vectorToString(stridesVec_) << ", axes:" << vectorToString(axesVec_);
    begin_ = aclCreateIntArray(beginVec_.data(), beginVec_.size());
    end_ = aclCreateIntArray(endVec_.data(), endVec_.size());
    strides_ = aclCreateIntArray(stridesVec_.data(), stridesVec_.size());
    axes_ = aclCreateIntArray(axesVec_.data(), axesVec_.size());
    return 0;
}

uint32_t SliceScatterOperation::GetInputNum() const { return NUM2; }

uint32_t SliceScatterOperation::GetOutputNum() const { return NUM1; }

int SliceScatterOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnStridedSliceAssignV2GetWorkspaceSize start";
    int ret = aclnnStridedSliceAssignV2GetWorkspaceSize(
        aclInTensors_.at(0).tensor, aclInTensors_.at(1).tensor, begin_, end_, strides_, axes_, &workspaceSize, &aclExecutor_);
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
    int64_t dim, start, end, step, rank;
    std::vector<int64_t> beginVec, endVec, stridesVec, axesVec;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("rank")) {
        rank = paramJson["rank"].get<std::int64_t>();
        axesVec.resize(rank);
        for (size_t i = 0; i < rank; ++i) {
            axesVec[i] = i;
        }
    }
    if (paramJson.contains("dim")) {
        dim = paramJson["dim"].get<std::int64_t>();
    }
    if (paramJson.contains("start")) {
        start = paramJson["start"].get<std::int64_t>();
        beginVec.resize(rank);
        for (size_t i = 0; i < rank; ++i) {
            beginVec[i] = i == dim ? start : 0;
        }
    }
    if (paramJson.contains("end")) {
        end = paramJson["end"].get<std::int64_t>();
        endVec.resize(rank);
        for (size_t i = 0; i < rank; ++i) {
            endVec[i] = i == dim ? end : -1;
        }
    }
    if (paramJson.contains("step")) {
        step = paramJson["step"].get<std::int64_t>();
        stridesVec.resize(rank);
        for (size_t i = 0; i < rank; ++i) {
            stridesVec[i] = i == dim ? step : 1;
        }
    }
    DICP_LOG(INFO) << "SliceScatterOperationCreate: name: " << opName << ", begin:" << vectorToString(beginVec) << ", end:" << vectorToString(endVec)
                   << ", strides:" << vectorToString(stridesVec);
    atb::Operation* op = new SliceScatterOperation(opName, beginVec, endVec, stridesVec, axesVec);
    return op;
}

REGISTER_OPERATION(SliceScatterOperation, SliceScatterOperationCreate);

}  // namespace dicp
