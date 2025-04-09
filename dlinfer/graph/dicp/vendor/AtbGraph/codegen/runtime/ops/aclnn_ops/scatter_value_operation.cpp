#include "scatter_value_operation.h"

#include "aclnnop/aclnn_scatter.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnScatterValueOperation::AclNnScatterValueOperation(const std::string& name, int64_t dim, float value, const std::string& value_dtype, int64_t reduce) : AclNnOperation(name), dim_(dim), reduce_(reduce) {
    value_ = DICPScalar(value, value_dtype);
    aclValue_ = aclCreateScalar(value_.getValuePtr(), value_.getDataType());
}

AclNnScatterValueOperation::~AclNnScatterValueOperation() {
    if (aclValue_ != nullptr) {
        aclDestroyScalar(aclValue_);
    }
}

atb::Status AclNnScatterValueOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnScatterValueOperation::GetInputNum() const { return NUM2; }

uint32_t AclNnScatterValueOperation::GetOutputNum() const { return NUM1; }

int AclNnScatterValueOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnMulsGetWorkspaceSize start";

    int ret = aclnnScatterValueGetWorkspaceSize(aclInTensors_.at(0).tensor, dim_, aclInTensors_.at(1).tensor, aclValue_, reduce_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnScatterValueGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnScatterValueOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnMuls start";
    int ret = aclnnScatterValue(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnMuls end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnScatterValueOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t dim;
    int64_t reduce;
    float value;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("dim")) {
        dim = paramJson["dim"].get<int64_t>();
    }
    if (paramJson.contains("value")) {
        value = paramJson["value"].get<float>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    if (paramJson.contains("reduce")) {
        reduce = paramJson["reduce"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnScatterValueOperation: name: " << opName << " dim:" << dim << " value:" << value << " dtype:" << dtype << " reduce:" << reduce;
    atb::Operation* op = new AclNnScatterValueOperation(opName, dim, value, dtype, reduce);
    return op;
}

REGISTER_OPERATION(AclNnScatterValueOperation, AclNnScatterValueOperationCreate);

}  // namespace dicp
