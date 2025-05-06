#include "zeros_operation.h"

#include <cstdint>

#include "aclnnop/aclnn_zero.h"
#include "ops/operation_creator.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "utils/common.h"
#include "utils/global_dict.h"
#include "utils/log.h"

namespace dicp {

const int NUM0 = 0;
const int NUM1 = 1;

ZerosOperation::ZerosOperation(const std::string& name, const std::vector<std::string>& size, aclDataType dtype) : AclNnOperation(name), opName_(name) {
    dtype_ = dtype;
    size_.resize(size.size());
    for (size_t i = 0; i < size.size(); ++i) {
        bool is_dynamic = !std::isdigit(size[i][0]);
        if (is_dynamic) {
            dynamic_size_[i] = size[i];
        } else {
            size_[i] = std::stol(size[i]);
        }
    }
    has_dynamic_size_ = dynamic_size_.size() > 0;
}

ZerosOperation::~ZerosOperation() {}

std::string ZerosOperation::GetName() const { return opName_; }

atb::Status ZerosOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(0).shape.dimNum = size_.size();

    const auto* global_dict = has_dynamic_size_ ? &GetGlobalDictData() : nullptr;

    for (size_t i = 0; i < size_.size(); ++i) {
        if (has_dynamic_size_ && global_dict) {
            const auto dynamic_size = dynamic_size_.find(i);
            if (dynamic_size != dynamic_size_.end()) {
                const auto it = global_dict->find(dynamic_size->second);
                if (it != global_dict->end()) {
                    outTensorDescs.at(0).shape.dims[i] = it->second;
                } else {
                    DICP_LOG(ERROR) << "Cannot find key " << dynamic_size->second << " in global_dict";
                    outTensorDescs.at(0).shape.dims[i] = size_[i];  // Fallback to static size
                }
                continue;
            }
        }
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
    std::vector<std::string> size;
    aclDataType dtype = aclDataType::ACL_DT_UNDEFINED;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("size")) {
        size = std::move(paramJson["size"].get<std::vector<std::string>>());
    }
    if (paramJson.contains("outTensorType")) {
        dtype = static_cast<aclDataType>(paramJson["outTensorType"].get<int32_t>());
    }
    DICP_LOG(INFO) << "ZerosOperation: name: " << opName << " viewShape:" << vectorToString<std::string>(size);
    atb::Operation* op = new ZerosOperation(opName, size, dtype);
    return op;
}

REGISTER_OPERATION(ZerosOperation, ZerosOperationCreate);

}  // namespace dicp
