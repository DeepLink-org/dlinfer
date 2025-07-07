#include "fused_lora_operation.h"

#include <cstdint>
#include <unordered_set>
#include <algorithm>

#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_grouped_matmul_v4.h"
#include "aclnnop/aclnn_permute.h"
#include "ops/operation_creator.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "utils/common.h"
#include "utils/log.h"
#include "utils/scalar.h"

#include <atb/utils.h>

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM8 = 8;

CustomFusedLoraOperation::CustomFusedLoraOperation(const std::string& name, const std::string& dtype) : opName_(name), dtype_(dtype) {}

CustomFusedLoraOperation::~CustomFusedLoraOperation() {}

std::string CustomFusedLoraOperation::GetName() const { return opName_; }

atb::Status CustomFusedLoraOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    DICP_LOG(INFO) << opName_ << " infer shape start";
    const auto totalLen = inTensorDescs.at(0).shape.dims[0];
    const auto totalRanks = inTensorDescs.at(1).shape.dims[0];
    const auto loraBDim = inTensorDescs.at(2).shape.dims[1];

    // Main output tensor
    outTensorDescs.at(0).shape.dimNum = 2;
    outTensorDescs.at(0).shape.dims[0] = totalLen;
    outTensorDescs.at(0).shape.dims[1] = loraBDim;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;

    // Internal gemm(x, lora_a) output
    outTensorDescs.at(1).shape.dimNum = 2;
    outTensorDescs.at(1).shape.dims[0] = totalLen;
    outTensorDescs.at(1).shape.dims[1] = totalRanks * totalLen;  // assuem totalRank is the max rank
    outTensorDescs.at(1).format = inTensorDescs.at(0).format;
    outTensorDescs.at(1).dtype = inTensorDescs.at(0).dtype;

    // Internal lora_a transpose output
    outTensorDescs.at(2).shape.dimNum = 2;
    outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
    outTensorDescs.at(2).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];
    outTensorDescs.at(2).format = inTensorDescs.at(1).format;
    outTensorDescs.at(2).dtype = inTensorDescs.at(1).dtype;
    return 0;
}

uint32_t CustomFusedLoraOperation::GetInputNum() const { return NUM8; }

uint32_t CustomFusedLoraOperation::GetOutputNum() const { return NUM3; }

AclNnTensor CustomFusedLoraOperation::CreateTensor(const atb::Tensor& atbTensor) {
    AclNnTensor aclNnTensor;
    aclNnTensor.atbTensor = atbTensor;
    return aclNnTensor;
}

int CustomFusedLoraOperation::CreateAclTensors(const atb::VariantPack& variantPack) {
    DICP_LOG(INFO) << opName_ << " CreateAclTensor start";

    const size_t inTensorCount = variantPack.inTensors.size();
    const size_t outTensorCount = variantPack.outTensors.size();
    
    aclInTensors_.resize(inTensorCount);
    aclOutTensors_.resize(outTensorCount);
    
    for (size_t i = 0; i < inTensorCount; ++i) {
        aclInTensors_[i] = CreateTensor(variantPack.inTensors.at(i));
    }

    for (size_t i = 0; i < outTensorCount; ++i) {
        aclOutTensors_[i] = CreateTensor(variantPack.outTensors.at(i));
    }

    DICP_LOG(INFO) << opName_ << " CreateAclTensor end";
    return 0;
}

void CustomFusedLoraOperation::ClearAclScalrs() {
    for (auto* scalar : aclScalingScalar_) {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
        }
    }
    aclScalingScalar_.clear();
}

void CustomFusedLoraOperation::ClearInternal() {
    ClearAclScalrs();
    aclWeightA_.clear();
    aclWeightB_.clear();
    aclWeightATranspose_.clear();
    weightA_.clear();
    weightB_.clear();
    weightATranspose_.clear();    

    aclScalingInput_.clear();
    scalingInput_.clear();
    aclScalingWeight_.clear();
    scalingWeight_.clear();

    aclScalingWorkspace_.clear();
    aclScalingExecutor_.clear();
}

// Helper function to create weight tensor
atb::Tensor CustomFusedLoraOperation::CreateWeightTensor(const atb::Tensor& baseTensor, int64_t rank, int64_t dim, uint64_t offset) {
    atb::Tensor weightTensor;
    weightTensor.desc.dtype = baseTensor.desc.dtype;
    weightTensor.desc.format = baseTensor.desc.format;
    weightTensor.desc.shape.dimNum = baseTensor.desc.shape.dimNum;
    weightTensor.desc.shape.dims[0] = rank;
    weightTensor.desc.shape.dims[1] = dim;
    weightTensor.dataSize = atb::Utils::GetTensorSize(weightTensor.desc);
    weightTensor.deviceData = static_cast<uint8_t*>(baseTensor.deviceData) + offset;
    return weightTensor;
}

// Helper function to calculate offset for weight tensors
uint64_t CustomFusedLoraOperation::CalculateWeightOffset(const std::vector<int32_t>& ranksVec, size_t adapterId, uint64_t tensorSizePerRank) {
    uint64_t offset = 0;
    for (size_t j = 0; j < adapterId; ++j) {
        offset += tensorSizePerRank * static_cast<uint64_t>(ranksVec[j]);
    }
    return offset;
}

int CustomFusedLoraOperation::Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) {
    DICP_LOG(INFO) << opName_ << " Setup start";

    DICP_CHECK_RET(CreateAclTensors(variantPack));

    // Create tensors for non-host tensors
    const std::unordered_set<size_t> hostTensorIndices{3, 4, 6};
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        if (hostTensorIndices.find(i) == hostTensorIndices.end()) {
            aclInTensors_.at(i).CreateTensor(opName_);
        }
    }

    for (auto& outTensor : aclOutTensors_) {
        outTensor.CreateTensor(opName_);
    }

    // Input tensor mapping:
    // 0: x (input tensor)
    // 1: lora_a (LoRA A weights)
    // 2: lora_b (LoRA B weights)
    // 3: scaling
    // 4: ranks (host tensor)
    // 5: seq_lens
    // 6: adapter_ids (host tensor)
    // 7 seq_lens_cpu (host tensor)

    // Extract host data
    const int32_t* ranksPtr = static_cast<const int32_t*>(variantPack.inTensors.at(4).hostData);
    const int32_t* adapterIdsPtr = static_cast<const int32_t*>(variantPack.inTensors.at(6).hostData);
    const int32_t* seqLensPtr = static_cast<const int32_t*>(variantPack.inTensors.at(7).hostData);

    const size_t ranksCount = variantPack.inTensors.at(4).desc.shape.dims[0];
    const size_t adapterIdsCount = variantPack.inTensors.at(6).desc.shape.dims[0];
    const size_t seqLensCount = variantPack.inTensors.at(7).desc.shape.dims[0];

    std::vector<int32_t> ranksVec(ranksPtr, ranksPtr + ranksCount);
    std::vector<int32_t> adapterIdsVec(adapterIdsPtr, adapterIdsPtr + adapterIdsCount);
    std::vector<int32_t> seqLensVec(seqLensPtr, seqLensPtr + seqLensCount);

    ranksVec[0] = 1;

    const int64_t loraADim = variantPack.inTensors.at(1).desc.shape.dims[1];
    const int64_t loraBDim = variantPack.inTensors.at(2).desc.shape.dims[1];

    ClearInternal();
    
    // Pre-allocate vectors to avoid reallocations
    weightA_.reserve(adapterIdsVec.size());
    weightATranspose_.reserve(adapterIdsVec.size());
    weightB_.reserve(adapterIdsVec.size());

    aclWeightA_.reserve(adapterIdsVec.size());
    aclWeightB_.reserve(adapterIdsVec.size());
    aclWeightATranspose_.reserve(adapterIdsVec.size());

    scalingWeight_.reserve(adapterIdsVec.size());
    scalingInput_.reserve(adapterIdsVec.size());
    aclScalingWeight_.reserve(adapterIdsVec.size());
    aclScalingInput_.reserve(adapterIdsVec.size());


    bool singleInfer = adapterIdsVec.size() == 1;
    int32_t totalRanks = 0;

    // Create weight tensors for each adapter
    for (size_t i = 0; i < adapterIdsVec.size(); ++i) {
        const int32_t adapterId = adapterIdsVec[i];
        const int32_t rank = ranksVec[adapterId];
        totalRanks += rank;

        // Create LoRA A weight tensor
        atb::Tensor weightA;
        weightA.desc.dtype = variantPack.inTensors.at(1).desc.dtype;
        weightA.desc.format = variantPack.inTensors.at(1).desc.format;
        if (singleInfer) {
            weightA.desc.shape.dimNum = 3;
            weightA.desc.shape.dims[0] = 1;
            weightA.desc.shape.dims[1] = rank;
            weightA.desc.shape.dims[2] = loraADim;

        } else {
            weightA.desc.shape.dimNum = 2;
            weightA.desc.shape.dims[0] = rank;
            weightA.desc.shape.dims[1] = loraADim;
        }
        const uint64_t weightASize = atb::Utils::GetTensorSize(weightA.desc);
        const uint64_t loraASizePerRank = weightASize / rank;
        const uint64_t offsetA = CalculateWeightOffset(ranksVec, adapterId, loraASizePerRank);
        weightA.deviceData = static_cast<uint8_t*>(variantPack.inTensors.at(1).deviceData) + offsetA;

        auto aclnnWeightA = CreateTensor(weightA);
        aclnnWeightA.CreateTensor(opName_);
        aclWeightA_.push_back(aclnnWeightA);

        atb::Tensor weightATranspose;
        weightATranspose.desc.dtype = variantPack.inTensors.at(1).desc.dtype;
        weightATranspose.desc.format = variantPack.inTensors.at(1).desc.format;
        if (singleInfer) {
            weightATranspose.desc.shape.dimNum = 3;
            weightATranspose.desc.shape.dims[0] = 1;
            weightATranspose.desc.shape.dims[1] = loraADim;
            weightATranspose.desc.shape.dims[2] = rank;
        } else {
            weightATranspose.desc.shape.dimNum = 2;
            weightATranspose.desc.shape.dims[0] = loraADim;
            weightATranspose.desc.shape.dims[1] = rank;
        }
        weightATranspose.deviceData = static_cast<uint8_t*>(variantPack.outTensors.at(2).deviceData) + offsetA;

        auto aclnnWeightATranspose = CreateTensor(weightATranspose);
        aclnnWeightATranspose.CreateTensor(opName_);
        aclWeightATranspose_.push_back(aclnnWeightATranspose);

        weightATransposeIdMap_[adapterId] = aclWeightATranspose_.size() - 1;

        // Create LoRA B weight tensor
        atb::Tensor weightB;
        weightB.desc.dtype = variantPack.inTensors.at(2).desc.dtype;
        weightB.desc.format = variantPack.inTensors.at(2).desc.format;
        if (singleInfer) {
            weightB.desc.shape.dimNum = 3;
            weightB.desc.shape.dims[0] = 1;
            weightB.desc.shape.dims[1] = rank;
            weightB.desc.shape.dims[2] = loraBDim;
        } else {
            weightB.desc.shape.dimNum = 2;
            weightB.desc.shape.dims[0] = rank;
            weightB.desc.shape.dims[1] = loraBDim;
        }
        const uint64_t weightBSize = atb::Utils::GetTensorSize(weightB.desc);
        const uint64_t loraBSizePerRank = weightBSize / rank;
        const uint64_t offsetB = CalculateWeightOffset(ranksVec, adapterId, loraBSizePerRank);
        weightB.deviceData = static_cast<uint8_t*>(variantPack.inTensors.at(2).deviceData) + offsetB;

        auto aclnnWeightB = CreateTensor(weightB);
        aclnnWeightB.CreateTensor(opName_);
        aclWeightB_.push_back(aclnnWeightB);
    }

    // transpose weight A
    std::vector<int64_t> permuteDims;
    if (singleInfer) {
        permuteDims = {0, 2, 1};
    } else {
        permuteDims = {1, 0};
    }
    aclIntArray *permuteDimsArray = aclCreateIntArray(permuteDims.data(), permuteDims.size());
    for (const auto& [adapterId, weightATransposeIndex] : weightATransposeIdMap_) {
        aclWeightAPermuteExecutor_[adapterId] = nullptr;
        aclWeightAPermuteWorkspace_[adapterId] = 0;

        auto& weightA = aclWeightA_[weightATransposeIndex];
        auto& weightATranspose = aclWeightATranspose_[weightATransposeIndex];


        int ret = aclnnPermuteGetWorkspaceSize(weightA.tensor,
                                           permuteDimsArray,
                                           weightATranspose.tensor,
                                           &aclWeightAPermuteWorkspace_[adapterId],
                                           &aclWeightAPermuteExecutor_[adapterId]);
        DICP_LOG(INFO) << opName_ << " aclnnPermuteGetWorkspaceSize size[" << adapterId << "]: " << aclWeightAPermuteWorkspace_[adapterId] << ", ret: " << ret;
    }

    // Setup grouped matrix multiplication
    DICP_LOG(INFO) << opName_ << " Setting up grouped matrix multiplication";
    
    // Create input tensor list
    std::vector<aclTensor*> xTmp;
    if (singleInfer) {
        xTmp = {aclInTensors_.at(0).tensor};
    } else {
        xTmp.reserve(seqLensVec.size());
        // split input by seq len
        for (size_t i = 0; i < seqLensVec.size(); ++i) {
            atb::Tensor slicedInput;
            slicedInput.desc.dtype = aclInTensors_.at(0).atbTensor.desc.dtype;
            slicedInput.desc.format = aclInTensors_.at(0).atbTensor.desc.format;
            slicedInput.desc.shape.dimNum = aclInTensors_.at(0).atbTensor.desc.shape.dimNum;
            slicedInput.desc.shape.dims[0] = seqLensVec[i];
            slicedInput.desc.shape.dims[1] = aclInTensors_.at(0).atbTensor.desc.shape.dims[1];  
            slicedInput.dataSize = atb::Utils::GetTensorSize(slicedInput.desc);

            auto offset = CalculateWeightOffset(seqLensVec, i, slicedInput.dataSize / seqLensVec[i]);
            slicedInput.deviceData = static_cast<uint8_t*>(aclInTensors_.at(0).atbTensor.deviceData) + offset;
            auto aclnnSlicedInput = CreateTensor(slicedInput);
            aclnnSlicedInput.CreateTensor(opName_);
            xTmp.push_back(aclnnSlicedInput.tensor);
        }
    }
    aclTensorList* xTensorList = aclCreateTensorList(xTmp.data(), xTmp.size());
    if (!xTensorList) {
        DICP_LOG(ERROR) << opName_ << " Failed to create x tensor list";
        return -1;
    }

    // Create weight tensor lists
    std::vector<aclTensor*> weightTmpA;
    std::vector<aclTensor*> weightTmpB;
    weightTmpA.reserve(aclWeightATranspose_.size());
    weightTmpB.reserve(aclWeightB_.size());
    
    for (const auto& weight : aclWeightATranspose_) {
        weightTmpA.push_back(weight.tensor);
    }
    for (const auto& weight : aclWeightB_) {
        weightTmpB.push_back(weight.tensor);
    }
    
    aclTensorList* weightTensorListA = aclCreateTensorList(weightTmpA.data(), weightTmpA.size());
    aclTensorList* weightTensorListB = aclCreateTensorList(weightTmpB.data(), weightTmpB.size());

    if (!weightTensorListA || !weightTensorListB) {
        DICP_LOG(ERROR) << opName_ << " Failed to create weight tensor lists";
        return -1;
    }

    // Create output tensor lists
    // slice aclOutTensors_.at(1)
    std::vector<aclTensor*> loraATmp;
    if (singleInfer) {
        atb::Tensor loraASliceOutput;
        loraASliceOutput.desc.dtype = aclOutTensors_.at(1).atbTensor.desc.dtype;
        loraASliceOutput.desc.format = aclOutTensors_.at(1).atbTensor.desc.format;
        loraASliceOutput.desc.shape.dimNum = aclOutTensors_.at(1).atbTensor.desc.shape.dimNum;
        loraASliceOutput.desc.shape.dims[0] = aclOutTensors_.at(1).atbTensor.desc.shape.dims[0];
        loraASliceOutput.desc.shape.dims[1] = totalRanks / adapterIdsVec.size();  
        loraASliceOutput.dataSize = atb::Utils::GetTensorSize(loraASliceOutput.desc);
        loraASliceOutput.deviceData = aclOutTensors_.at(1).atbTensor.deviceData;
        auto aclnnLoraASliceOutput = CreateTensor(loraASliceOutput);
        aclnnLoraASliceOutput.CreateTensor(opName_);
        loraATmp = {aclnnLoraASliceOutput.tensor};
    } else {
        loraATmp.reserve(seqLensVec.size());
        // split input by seq len
        for (size_t i = 0; i < seqLensVec.size(); ++i) {
            atb::Tensor slicedOutput;
            slicedOutput.desc.dtype = aclOutTensors_.at(1).atbTensor.desc.dtype;
            slicedOutput.desc.format = aclOutTensors_.at(1).atbTensor.desc.format;
            slicedOutput.desc.shape.dimNum = aclOutTensors_.at(1).atbTensor.desc.shape.dimNum;
            slicedOutput.desc.shape.dims[0] = seqLensVec[i];
            slicedOutput.desc.shape.dims[1] = ranksVec[adapterIdsVec[i]];  
            slicedOutput.dataSize = atb::Utils::GetTensorSize(slicedOutput.desc);

            auto offset = CalculateWeightOffset(seqLensVec, i, slicedOutput.dataSize / seqLensVec[i]);
            slicedOutput.deviceData = static_cast<uint8_t*>(aclOutTensors_.at(1).atbTensor.deviceData) + offset;
            auto aclnnSlicedOutput = CreateTensor(slicedOutput);
            aclnnSlicedOutput.CreateTensor(opName_);
            loraATmp.push_back(aclnnSlicedOutput.tensor);
        }
    }

    std::vector<aclTensor*> loraBTmp{aclOutTensors_.at(0).tensor};

    aclTensorList* loraAOutTensorList = aclCreateTensorList(loraATmp.data(), loraATmp.size());
    aclTensorList* loraBOutTensorList = aclCreateTensorList(loraBTmp.data(), loraBTmp.size());

    if (!loraAOutTensorList || !loraBOutTensorList) {
        DICP_LOG(ERROR) << opName_ << " Failed to create output tensor lists";
        return -1;
    }
    
    // Setup LoRA A grouped matrix multiplication
    int ret = aclnnGroupedMatmulV4GetWorkspaceSize(xTensorList,                                           // x
                                                   weightTensorListA,                                     // weight
                                                   nullptr,                                               // biasOptional
                                                   nullptr,                                               // scaleOptional
                                                   nullptr,                                               // offsetOptional
                                                   nullptr,                                               // antiquantScaleOptional
                                                   nullptr,                                               // antiquantOffsetOptional
                                                   nullptr,                                               // perTokenScaleOptional
                                                   singleInfer ? aclInTensors_.at(5).tensor : nullptr,    // groupListOptional
                                                   nullptr,                                               // activationInputOptional
                                                   nullptr,                                               // activationQuantScaleOptional
                                                   nullptr,                                               // activationQuantOffsetOptional
                                                   singleInfer ? 2 : 0,                                   // splitItem
                                                   singleInfer ? 0 : -1,                                  // groupType
                                                   1,                                                     // groupListType
                                                   0,                                                     // actType
                                                   loraAOutTensorList,                                    // out
                                                   nullptr,                                               // activationFeatureOutOptional
                                                   nullptr,                                               // dynQuantScaleOutOptional
                                                   &loraAGroupedGemmWorkspace_,
                                                   &aclLoraAGroupedGemmExecutor_);
    DICP_LOG(INFO) << opName_ << " LoRA A grouped matmul workspace size: " << loraAGroupedGemmWorkspace_ << ", ret: " << ret;

    // Setup LoRA B grouped matrix multiplication
    ret = aclnnGroupedMatmulV4GetWorkspaceSize(loraAOutTensorList,                                // x
                                               weightTensorListB,                                 // weight
                                               nullptr,                                           // biasOptional
                                               nullptr,                                           // scaleOptional
                                               nullptr,                                           // offsetOptional
                                               nullptr,                                           // antiquantScaleOptional
                                               nullptr,                                           // antiquantOffsetOptional
                                               nullptr,                                           // perTokenScaleOptional
                                               aclInTensors_.at(5).tensor,                        // groupListOptional
                                               nullptr,                                           // activationInputOptional
                                               nullptr,                                           // activationQuantScaleOptional
                                               nullptr,                                           // activationQuantOffsetOptional
                                               2,                                                 // splitItem
                                               0,                                                 // groupType
                                               1,                                                 // groupListType
                                               0,                                                 // actType
                                               loraBOutTensorList,                                // out
                                               nullptr,                                           // activationFeatureOutOptional
                                               nullptr,                                           // dynQuantScaleOutOptional
                                               &loraBGroupedGemmWorkspace_,
                                               &aclLoraBGroupedGemmExecutor_);
    DICP_LOG(INFO) << opName_ << " LoRA B grouped matmul workspace size: " << loraBGroupedGemmWorkspace_ << ", ret: " << ret;
    
    // Setup scaling operations
    aclScalingWorkspace_.resize(adapterIdsVec.size());
    aclScalingExecutor_.resize(adapterIdsVec.size());
    
    for (size_t i = 0; i < adapterIdsVec.size(); ++i) {
        const int32_t adapterId = adapterIdsVec[i];
        const auto& inputAtbTensor = aclOutTensors_.at(0).atbTensor;
        const auto& scalingAtbTensor = aclInTensors_.at(3).atbTensor;

        // Create slice tensor for scaling
        atb::Tensor input;
        input.desc.dtype = inputAtbTensor.desc.dtype;
        input.desc.format = inputAtbTensor.desc.format;
        input.desc.shape.dimNum = inputAtbTensor.desc.shape.dimNum;
        input.desc.shape.dims[0] = seqLensVec[i];
        input.desc.shape.dims[1] = loraBDim;
        input.dataSize = atb::Utils::GetTensorSize(input.desc);

        uint64_t offset = 0;
        for (size_t j = 0; j < i; ++j) {
            offset += loraBDim * static_cast<uint64_t>(seqLensVec[j]);
        }
        input.deviceData = static_cast<uint8_t*>(inputAtbTensor.deviceData) + offset;

        scalingInput_.push_back(input);

        // create slice tensor for scaling weight
        atb::Tensor weight;
        weight.desc.dtype = scalingAtbTensor.desc.dtype;
        weight.desc.format = scalingAtbTensor.desc.format;
        weight.desc.shape.dimNum = 1;
        weight.desc.shape.dims[0] = 1;
        weight.dataSize = atb::Utils::GetTensorSize(weight.desc);

        const uint64_t weight_offset = weight.dataSize * adapterId;
        weight.deviceData = static_cast<uint8_t*>(scalingAtbTensor.deviceData) + weight_offset;

        scalingWeight_.push_back(weight);

        auto aclnnScalingInput = CreateTensor(input);
        aclnnScalingInput.CreateTensor(opName_);
        aclScalingInput_.push_back(aclnnScalingInput);

        auto aclnnScalingWeight = CreateTensor(weight);
        aclnnScalingWeight.CreateTensor(opName_);
        aclScalingWeight_.push_back(aclnnScalingWeight);

        ret = aclnnInplaceMulGetWorkspaceSize(aclScalingInput_.back().tensor,
                                               aclScalingWeight_.back().tensor,
                                               &aclScalingWorkspace_[i],
                                               &aclScalingExecutor_[i]);
        DICP_LOG(INFO) << opName_ << " Scaling workspace size[" << i << "]: " << aclScalingWorkspace_[i] << ", ret: " << ret;
    }

    // Calculate total workspace size
    const uint64_t scalingMaxValue = aclScalingWorkspace_.empty() ? 0 : *std::max_element(aclScalingWorkspace_.begin(), aclScalingWorkspace_.end());
    workspaceSize = std::max({loraAGroupedGemmWorkspace_, loraBGroupedGemmWorkspace_, scalingMaxValue});

    DICP_LOG(INFO) << opName_ << " Setup completed, total workspace size: " << workspaceSize;
    return ret;
}

int CustomFusedLoraOperation::Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) {
    DICP_LOG(INFO) << opName_ << " execute start";
    if (!context) {
        DICP_LOG(ERROR) << opName_ << " execute fail, context param is null";
        return atb::ERROR_INVALID_PARAM;
    }

    aclrtStream stream = context->GetExecuteStream();
    if (!stream) {
        DICP_LOG(ERROR) << opName_ << " execute fail, execute stream in context is null";
        return atb::ERROR_INVALID_PARAM;
    }

    // transpose weightA
    for (const auto& [adapterId, weightATransposeIndex] : weightATransposeIdMap_) {
        int ret = aclnnPermute(workspace, aclWeightAPermuteWorkspace_[adapterId], aclWeightAPermuteExecutor_[adapterId], stream);
        DICP_LOG(INFO) << opName_ << " aclnnPermute completed, ret: " << ret;
    }

    // Execute LoRA A grouped matrix multiplication
    int ret = aclnnGroupedMatmulV4(workspace, loraAGroupedGemmWorkspace_, aclLoraAGroupedGemmExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " LoRA A grouped matmul completed, ret: " << ret;

    // Execute LoRA B grouped matrix multiplication
    ret = aclnnGroupedMatmulV4(workspace, loraBGroupedGemmWorkspace_, aclLoraBGroupedGemmExecutor_, stream);
    DICP_LOG(INFO) << opName_ << " LoRA B grouped matmul completed, ret: " << ret;

    // Execute scaling operations
    for (size_t i = 0; i < aclScalingExecutor_.size(); ++i) {
        ret = aclnnInplaceMul(workspace, aclScalingWorkspace_[i], aclScalingExecutor_[i], stream);
        DICP_LOG(INFO) << opName_ << " Scaling operation[" << i << "] completed, ret: " << ret;
    }

    DICP_LOG(INFO) << opName_ << " execute end";
    return 0;
}

atb::Operation* CustomFusedLoraOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    std::string dtype;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("dtype")) {
        dtype = paramJson["dtype"].get<std::string>();
    }
    DICP_LOG(INFO) << "CustomFusedLoraOperationCreate: name: " << opName << " dtype:" << dtype;
    atb::Operation* op = new CustomFusedLoraOperation(opName, dtype);
    return op;
}

REGISTER_OPERATION(CustomFusedLoraOperation, CustomFusedLoraOperationCreate);

}  // namespace dicp
