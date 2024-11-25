#include "arange_operation.h"

#include <cmath>

#include "aclnnop/aclnn_arange.h"
#include "utils/log.h"

namespace dicp {

const int NUM0 = 0;
const int NUM1 = 1;

AclNnArangeOperation::AclNnArangeOperation(const std::string& name, int64_t start, int64_t end, int64_t step)
    : AclNnOperation(name), start_(start), end_(end), step_(step) {
    aclStart_ = aclCreateScalar(&start_, aclDataType::ACL_INT64);
    aclEnd_ = aclCreateScalar(&end_, aclDataType::ACL_INT64);
    aclStep_ = aclCreateScalar(&step_, aclDataType::ACL_INT64);
    auto sizeArange = std::ceil(static_cast<double>(end_ - start_) / step_);
    sizeArange_ = static_cast<int64_t>(sizeArange);
}

AclNnArangeOperation::~AclNnArangeOperation() {
    if (aclStart_ != nullptr) {
        aclDestroyScalar(aclStart_);
    }
    if (aclEnd_ != nullptr) {
        aclDestroyScalar(aclEnd_);
    }
    if (aclStep_ != nullptr) {
        aclDestroyScalar(aclStep_);
    }
}

atb::Status AclNnArangeOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(0).shape.dimNum = NUM1;
    outTensorDescs.at(0).dtype = aclDataType::ACL_INT64;
    outTensorDescs.at(0).shape.dims[0] = sizeArange_;
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnArangeOperation::GetInputNum() const { return NUM0; }

uint32_t AclNnArangeOperation::GetOutputNum() const { return NUM1; }

int AclNnArangeOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnArangeGetWorkspaceSize start";

    int ret = aclnnArangeGetWorkspaceSize(aclStart_, aclEnd_, aclStep_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnArangeGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnArangeOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnArange start";
    int ret = aclnnArange(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnArange end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnArangeOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t start = 0;
    int64_t end = 0;
    int64_t step = 0;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("start")) {
        start = paramJson["start"].get<int64_t>();
    }
    if (paramJson.contains("end")) {
        end = paramJson["end"].get<int64_t>();
    }
    if (paramJson.contains("step")) {
        step = paramJson["step"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnArangeOperation: name: " << opName << " start:" << start << " end:" << end << " step:" << step;
    atb::Operation* op = new AclNnArangeOperation(opName, start, end, step);
    return op;
}

REGISTER_OPERATION(AclNnArangeOperation, AclNnArangeOperationCreate);

}  // namespace dicp
