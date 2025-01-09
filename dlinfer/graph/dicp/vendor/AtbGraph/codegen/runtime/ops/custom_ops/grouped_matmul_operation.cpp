// #include "grouped_matmul_operation.h"

// #include "aclnnop/aclnn_grouped_matmul_v3.h"
// #include "aclnnop/aclnn_permute.h"
// #include "utils/common.h"
// #include "utils/log.h"

// namespace dicp {

// const int NUM1 = 1;
// const int NUM2 = 2;
// const int NUM3 = 3;

// AclNnGroupedMatmulOperation::AclNnGroupedMatmulOperation(const std::string& name, int64_t splitItem) : opName_(name), splitItem_(splitItem) {}

// AclNnGroupedMatmulOperation::~AclNnGroupedMatmulOperation() {
//     for (size_t i = 0; i < aclInTensors_.size(); ++i) {
//         aclDestroyTensor(aclInTensors_[i].tensor);
//     }
//     aclInTensors_.clear();
//     for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
//         aclDestroyTensor(aclOutTensors_[i].tensor);
//     }
//     aclOutTensors_.clear();
// }

// std::string AclNnGroupedMatmulOperation::GetName() const { return opName_; }

// atb::Status AclNnGroupedMatmulOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const
// {
//     DICP_LOG(INFO) << opName_ << " infer shape start";

//     outTensorDescs.at(0).format = inTensorDescs.at(1).format;
//     outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(1).shape.dimNum;
//     outTensorDescs.at(0).dtype = inTensorDescs.at(1).dtype;
//     outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
//     outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(1).shape.dims[2];
//     outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[1];

//     outTensorDescs.at(1).format = inTensorDescs.at(0).format;
//     outTensorDescs.at(1).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
//     outTensorDescs.at(1).dtype = inTensorDescs.at(0).dtype;
//     outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
//     outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];

//     DICP_LOG(INFO) << opName_ << " infer shape end";
//     return 0;
// }

// uint32_t AclNnGroupedMatmulOperation::GetInputNum() const { return NUM3; }

// uint32_t AclNnGroupedMatmulOperation::GetOutputNum() const { return NUM2; }

// AclNnTensor AclNnGroupedMatmulOperation::CreateTensor(atb::Tensor atbTensor) {
//     AclNnTensor aclNnTensor;
//     aclNnTensor.atbTensor = atbTensor;
//     return aclNnTensor;
// }

// int AclNnGroupedMatmulOperation::CreateAclTensors(const atb::VariantPack& variantPack) {
//     DICP_LOG(INFO) << opName_ << " CreateAclTensor start";

//     aclInTensors_.resize(variantPack.inTensors.size());
//     for (size_t i = 0; i < aclInTensors_.size(); ++i) {
//         aclInTensors_[i] = CreateTensor(variantPack.inTensors.at(i));
//     }

//     aclOutTensors_.resize(variantPack.outTensors.size());
//     for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
//         aclOutTensors_[i] = CreateTensor(variantPack.outTensors.at(i));
//     }

//     DICP_LOG(INFO) << opName_ << " CreateAclTensor end";
//     return 0;
// }

// int AclNnGroupedMatmulOperation::Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) {
//     DICP_LOG(INFO) << opName_ << " aclnnGroupedMatmulGetWorkspaceSize start";

//     if (context == nullptr) {
//         DICP_LOG(ERROR) << opName_ << " setup context is null";
//         return atb::ERROR_INVALID_PARAM;
//     }

//     DICP_CHECK_RET(CreateAclTensors(variantPack));

//     for (size_t i = 0; i < aclInTensors_.size(); ++i) {
//         aclInTensors_.at(i).CreateTensor(opName_);
//     }

//     for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
//         aclOutTensors_.at(i).CreateTensor(opName_);
//     }

//     // permute
//     DICP_LOG(INFO) << opName_ << " aclnnPermuteGetWorkspaceSize start";
//     std::vector<int64_t> permuteDimsValue{0, 2, 1};
//     aclIntArray* permuteDims = aclCreateIntArray(permuteDimsValue.data(), permuteDimsValue.size());
//     int ret = aclnnPermuteGetWorkspaceSize(aclInTensors_.at(1).tensor, permuteDims, aclOutTensors_.at(0).tensor, &permuteWorkspaceSize_,
//     &aclPermuteExecutor_); workspaceSize = permuteWorkspaceSize_; DICP_LOG(INFO) << opName_ << " aclnnPermuteGetWorkspaceSize end, ret:" << ret << ",
//     workspaceSize:" << permuteWorkspaceSize_
//                    << ", aclExecutor:" << aclPermuteExecutor_;

//     // groupedMatmulV3
//     DICP_LOG(INFO) << opName_ << " aclnnGroupedMatmulGetWorkspaceSize start";
//     std::vector<aclTensor*> xTmp{aclInTensors_.at(0).tensor};
//     // aclTensorList* xTensorList = aclCreateTensorList(xTmp.data(), xTmp.size());
//     aclInTensorList_.push_back(aclCreateTensorList(xTmp.data(), xTmp.size()));
//     std::vector<aclTensor*> weightTmp{aclOutTensors_.at(0).tensor};
//     // aclTensorList* weightTensorList = aclCreateTensorList(weightTmp.data(), 64);
//     aclOutTensorList_.push_back(aclCreateTensorList(weightTmp.data(), weightTmp.size()));
//     std::vector<aclTensor*> outTmp{aclOutTensors_.at(1).tensor};
//     // aclTensorList* outTensorList = aclCreateTensorList(outTmp.data(), outTmp.size());
//     aclOutTensorList_.push_back(aclCreateTensorList(outTmp.data(), outTmp.size()));
//     DICP_LOG(INFO) << "aclnnGroupedMatmulGetWorkspaceSize create list end";
//     DICP_LOG(INFO) << aclInTensors_.size() << " ** " << aclOutTensors_.size();
//     ret = aclnnGroupedMatmulV3GetWorkspaceSize(aclInTensorList_.at(0),
//                                                aclOutTensorList_.at(0),
//                                                nullptr,
//                                                nullptr,
//                                                nullptr,
//                                                nullptr,
//                                                nullptr,
//                                                aclInTensors_.at(2).tensor,
//                                                splitItem_,
//                                                0,
//                                                aclOutTensorList_.at(1),
//                                                &groupedMatmulWorkspaceSize_,
//                                                &aclGroupedMatmulExecutor_);
//     workspaceSize = groupedMatmulWorkspaceSize_ > workspaceSize ? groupedMatmulWorkspaceSize_ : workspaceSize;
//     DICP_LOG(INFO) << opName_ << " aclnnGroupedMatmulGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << groupedMatmulWorkspaceSize_
//                    << ", aclExecutor:" << aclGroupedMatmulExecutor_;

//     return ret;
// }

// int AclNnGroupedMatmulOperation::Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) {
//     DICP_LOG(INFO) << opName_ << " execute start";
//     if (!context) {
//         DICP_LOG(ERROR) << opName_ << " execute fail, context param is null";
//         return atb::ERROR_INVALID_PARAM;
//     }

//     aclrtStream stream = context->GetExecuteStream();
//     if (!stream) {
//         DICP_LOG(ERROR) << opName_ << " execute fail, execute stream in context is null";
//         return atb::ERROR_INVALID_PARAM;
//     }

//     // permute
//     aclInTensors_.at(1).atbTensor.deviceData = variantPack.inTensors.at(1).deviceData;
//     aclOutTensors_.at(0).atbTensor.deviceData = variantPack.outTensors.at(0).deviceData;
//     DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclPermuteExecutor_, opName_, 0, true));
//     DICP_CHECK_RET(aclOutTensors_.at(1).InitTensor(aclPermuteExecutor_, opName_, 1, false));
//     DICP_LOG(INFO) << opName_ << " aclnnPermute start";
//     int ret = aclnnPermute(workspace, permuteWorkspaceSize_, aclPermuteExecutor_, stream);
//     DICP_LOG(INFO) << opName_ << " aclnnPermute end, ret:" << ret;

//     // groupedMatmulV3
//     aclInTensors_.at(0).atbTensor.deviceData = variantPack.inTensors.at(0).deviceData;
//     aclInTensors_.at(2).atbTensor.deviceData = variantPack.inTensors.at(2).deviceData;
//     aclOutTensors_.at(1).atbTensor.deviceData = variantPack.outTensors.at(1).deviceData;
//     DICP_CHECK_RET(aclInTensors_.at(0).InitTensor(aclGroupedMatmulExecutor_, opName_, 0, true));
//     DICP_CHECK_RET(aclInTensors_.at(2).InitTensor(aclGroupedMatmulExecutor_, opName_, 2, true));
//     DICP_CHECK_RET(aclOutTensors_.at(0).InitTensor(aclGroupedMatmulExecutor_, opName_, 0, true));
//     DICP_CHECK_RET(aclOutTensors_.at(1).InitTensor(aclGroupedMatmulExecutor_, opName_, 1, false));
//     DICP_LOG(INFO) << opName_ << " aclnnGroupedMatmulV3 start";
//     ret = aclnnGroupedMatmulV3(workspace, groupedMatmulWorkspaceSize_, aclGroupedMatmulExecutor_, stream);
//     DICP_LOG(INFO) << opName_ << " aclnnGroupedMatmulV3 end, ret:" << ret;

//     DICP_LOG(INFO) << opName_ << " execute end";
//     return 0;
// }

// atb::Operation* AclNnGroupedMatmulOperationCreate(const nlohmann::json& paramJson) {
//     std::string opName;
//     int64_t splitItem;
//     if (paramJson.contains("name")) {
//         opName = paramJson["name"].get<std::string>();
//     }
//     if (paramJson.contains("splitItem")) {
//         splitItem = paramJson["splitItem"].get<int64_t>();
//     }
//     DICP_LOG(INFO) << "AclNnGroupedMatmulOperation: name: " << opName << " splitItem:" << splitItem;
//     atb::Operation* op = new AclNnGroupedMatmulOperation(opName, splitItem);
//     return op;
// }

// REGISTER_OPERATION(AclNnGroupedMatmulOperation, AclNnGroupedMatmulOperationCreate);

// }  // namespace dicp
