#include "reduce_sum_operation.h"

#include "aclnnop/aclnn_reduce_sum.h"
#include "utils/common.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;

AclNnReduceSumOperation::AclNnReduceSumOperation(const std::string& name, const std::vector<int64_t>& dims, bool keepDim, const std::string& dtype)
    : AclNnOperation(name), dims_(std::move(dims)), keepDim_(keepDim), dtype_(get_acl_dtype(dtype)) {
    aclDims_ = aclCreateIntArray(dims_.data(), dims_.size());
}

AclNnReduceSumOperation::~AclNnReduceSumOperation() {
    if (aclDims_) {
        aclDestroyIntArray(aclDims_);
    }
}

atb::Status AclNnReduceSumOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";

    if (keepDim_) {
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
        outTensorDescs.at(0).dtype = dtype_;
        for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
            if (std::find(dims_.begin(), dims_.end(), i) != dims_.end()) {
                outTensorDescs.at(0).shape.dims[i] = 1;
            } else {
                outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
            }
        }
    } else {
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum - dims_.size();
        outTensorDescs.at(0).dtype = dtype_;
        int ii = 0;
        for (size_t i = 0; i < inTensorDescs.at(0).shape.dimNum; ++i) {
            if (std::find(dims_.begin(), dims_.end(), i) == dims_.end()) {
                outTensorDescs.at(0).shape.dims[ii] = inTensorDescs.at(0).shape.dims[i];
                ii++;
            }
        }
    }

    DICP_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AclNnReduceSumOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnReduceSumOperation::GetOutputNum() const { return NUM1; }

int AclNnReduceSumOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    DICP_LOG(INFO) << opName_ << " aclnnReduceSumGetWorkspaceSize start";

    int ret =
        aclnnReduceSumGetWorkspaceSize(aclInTensors_.at(0).tensor, aclDims_, keepDim_, dtype_, aclOutTensors_.at(0).tensor, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnReduceSumGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;

    return ret;
}

int AclNnReduceSumOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    DICP_LOG(INFO) << opName_ << " aclnnReduceSum start";
    int ret = aclnnReduceSum(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnReduceSum end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnReduceSumOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::vector<int64_t> dims;
    bool keepDim;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("dims")) {
        dims = paramJson["dims"].get<std::vector<int64_t>>();
    }
    if (paramJson.contains("keepDim")) {
        keepDim = paramJson["keepDim"].get<bool>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnReduceSumOperation: name: " << opName << " dims:" << vectorToString(dims) << " keepDim:" << keepDim << " dtype:" << dtype;
    atb::Operation* op = new AclNnReduceSumOperation(opName, dims, keepDim, dtype);
    return op;
}

REGISTER_OPERATION(AclNnReduceSumOperation, AclNnReduceSumOperationCreate);

}  // namespace dicp
