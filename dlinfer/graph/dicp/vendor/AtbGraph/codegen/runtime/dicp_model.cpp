

#include "dicp_model.h"

#include <acl/acl.h>
#include <atb/utils.h>
#include <torch/torch.h>

#include "model.h"
#include "utils/log.h"
#include "utils/misc.h"
#include "utils/tensor_utils.h"

using namespace dicp;

DICPModel::DICPModel(const std::string& modelPath) : modelPath_(modelPath) {
    modelId_ = utils::GetNewModelId();
    DICP_LOG(INFO) << "DICPModel create start, modelId:" << modelId_ << ", modelPath:" << modelPath_;
    model_ = std::make_shared<Model>(std::to_string(modelId_), modelPath);

    atb::Context* rawContext = nullptr;
    auto st = atb::CreateContext(&rawContext);
    DICP_LOG_IF(st != atb::NO_ERROR, ERROR) << "create atb context failed!";
    context_ = std::move(std::unique_ptr<atb::Context, decltype(&atb::DestroyContext)>(rawContext, atb::DestroyContext));
}

DICPModel::~DICPModel() { context_.reset(); };

void DICPModel::ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors, const std::string& param) {
    context_->SetExecuteStream(utils::GetCurrentStream());

    std::vector<atb::Tensor> inTensors;
    tensor_utils::TransferAtTensor2AtbTensor(atInTensors, inTensors);

    std::vector<atb::Tensor> outTensors;
    tensor_utils::TransferAtTensor2AtbTensor(atOutTensors, outTensors);

    model_->Execute(context_.get(), inTensors, outTensors, param);
}

TORCH_LIBRARY(DICPModel, m) { m.class_<DICPModel>("DICPModel").def(torch::init<std::string>()).def("execute_out", &DICPModel::ExecuteOut); }
