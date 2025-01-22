#include "zeros_operation.h"

#include <cstdint>

#include "aclnnop/aclnn_zero.h"
#include "ops/operation_creator.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "utils/common.h"
#include "utils/log.h"

namespace dicp {

const int NUM0 = 0;
const int NUM1 = 1;

ZerosOperation::ZerosOperation(const std::string& name, const std::vector<int64_t>& size, aclDataType dtype)
    : AclNnOperation(name), opName_(name), size_(size), dtype_(dtype) {}

ZerosOperation::~ZerosOperation() {}

std::string ZerosOperation::GetName() const { return opName_; }

atb::Status ZerosOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(0).shape.dimNum = size_.size();
    for (size_t i = 0; i < size_.size(); ++i) {
        outTensorDescs.at(0).shape.dims[i] = size_[i];
    }

    outTensorDescs.at(0).dtype = dtype_;
    return 0;
}

uint32_t ZerosOperation::GetInputNum() const { return NUM0; }

uint32_t ZerosOperation::GetOutputNum() const { return NUM1; }

int ZerosOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " ZerosOperationGetWorkspaceSize start";

    int ret = aclnnInplaceZeroGetWorkspaceSize(aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " ZerosOperationGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int ZerosOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " ZerosOperation start";
    int ret = aclnnInplaceZero(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " ZerosOperation end, ret:" << ret;
    return ret;
}

atb::Operation* ZerosOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::vector<int64_t> size;
    aclDataType dtype = aclDataType::ACL_DT_UNDEFINED;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("size")) {
        size = std::move(paramJson["size"].get<std::vector<int64_t>>());
    }
    if (paramJson.contains("outTensorType")) {
        dtype = static_cast<aclDataType>(paramJson["outTensorType"].get<int32_t>());
    }
    DICP_LOG(INFO) << "ZerosOperation: name: " << opName << " viewShape:" << vectorToString<int64_t>(size);
    atb::Operation* op = new ZerosOperation(opName, size, dtype);
    return op;
}

REGISTER_OPERATION(ZerosOperation, ZerosOperationCreate);

}  // namespace dicp
