#include "new_empty_operation.h"

#include <cstdint>

#include "aclnnop/aclnn_zero.h"
#include "ops/operation_creator.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "utils/common.h"
#include "utils/global_dict.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;

NewEmptyOperation::NewEmptyOperation(const std::string& name, const std::vector<std::string>& size) : AclNnOperation(name), opName_(name) {
    size_.resize(size.size());

    for (size_t i = 0; i < size.size(); ++i) {
        bool is_dynamic = !std::isdigit(size[i][0]);
        if (is_dynamic) {
            dynamic_size_[i] = size[i];
        } else {
            size_[i] = std::stol(size[i]);
        }
    }
}

NewEmptyOperation::~NewEmptyOperation() {}

std::string NewEmptyOperation::GetName() const { return opName_; }

atb::Status NewEmptyOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";

    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = size_.size();
    auto& global_dict = GetGlobalDictData();
    for (size_t i = 0; i < size_.size(); ++i) {
        auto dynamic_size = dynamic_size_.find(i);
        if (dynamic_size != dynamic_size_.end()) {
            auto it = global_dict.find(dynamic_size->second);
            if (it != global_dict.end()) {
                outTensorDescs.at(0).shape.dims[i] = static_cast<int64_t>(it->second);
            } else {
                DICP_LOG(ERROR) << "Cannot find key " << dynamic_size->second << " in global_dict";
            }
        } else {
            outTensorDescs.at(0).shape.dims[i] = size_[i];
        }
    }
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    return 0;
}

uint32_t NewEmptyOperation::GetInputNum() const { return NUM1; }

uint32_t NewEmptyOperation::GetOutputNum() const { return NUM1; }

int NewEmptyOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " NewEmptyOperationGetWorkspaceSize start";
    int ret = aclnnInplaceZeroGetWorkspaceSize(aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " NewEmptyOperationGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int NewEmptyOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnInplaceZero start";
    int ret = aclnnInplaceZero(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceZero end, ret:" << ret;
    return ret;
}

atb::Operation* NewEmptyOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::vector<std::string> size_str;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("size")) {
        size_str = paramJson["size"].get<std::vector<std::string>>();
    }
    DICP_LOG(INFO) << "ZerosOperation: name: " << opName << " viewShape:" << vectorToString<std::string>(size_str);
    atb::Operation* op = new NewEmptyOperation(opName, size_str);
    return op;
}

REGISTER_OPERATION(NewEmptyOperation, NewEmptyOperationCreate);

}  // namespace dicp
