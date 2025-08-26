#include "grouped_matmul_operation.h"

#include "aclnnop/aclnn_grouped_matmul_v3.h"
#include "ops/aclnn_ops/acl_nn_operation.h"
#include "utils/log.h"

namespace dicp {

const int NUM1 = 1;
const int NUM3 = 3;

AclNnGroupedMatmulOperation::AclNnGroupedMatmulOperation(const std::string& name, int64_t splitItem) : AclNnOperation(name) { this->splitItem = splitItem; }

AclNnGroupedMatmulOperation::~AclNnGroupedMatmulOperation() {}

atb::Status AclNnGroupedMatmulOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";

    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(1).shape.dims[2];

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnGroupedMatmulOperation::GetInputNum() const { return NUM3; }

uint32_t AclNnGroupedMatmulOperation::GetOutputNum() const { return NUM1; }

int AclNnGroupedMatmulOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnGroupedMatmulGetWorkspaceSize start";
    std::vector<aclTensor*> xTmp{aclInTensors_.at(0).tensor};
    aclTensorList* xTensorList = aclCreateTensorList(xTmp.data(), xTmp.size());
    std::vector<aclTensor*> weightTmp{aclInTensors_.at(1).tensor};
    aclTensorList* weightTensorList = aclCreateTensorList(weightTmp.data(), weightTmp.size());
    std::vector<aclTensor*> outTmp{aclOutTensors_.at(0).tensor};
    aclTensorList* outTensorList = aclCreateTensorList(outTmp.data(), outTmp.size());

    int ret = aclnnGroupedMatmulV3GetWorkspaceSize(xTensorList,
                                                   weightTensorList,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   aclInTensors_.at(2).tensor,
                                                   this->splitItem,
                                                   0,
                                                   outTensorList,
                                                   &workspaceSize,
                                                   &aclExecutor_);

    DICP_LOG(INFO) << opName_ << " aclnnGroupedMatmulGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize
                   << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnGroupedMatmulOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnGroupedMatmulV3 start";
    int ret = aclnnGroupedMatmulV3(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnGroupedMatmulV3 end, ret:" << ret;

    return 0;
}

atb::Operation* AclNnGroupedMatmulOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t splitItem;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("splitItem")) {
        splitItem = paramJson["splitItem"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnGroupedMatmulOperation: name: " << opName << " splitItem:" << splitItem;
    atb::Operation* op = new AclNnGroupedMatmulOperation(opName, splitItem);
    return op;
}

REGISTER_OPERATION(AclNnGroupedMatmulOperation, AclNnGroupedMatmulOperationCreate);

}  // namespace dicp
