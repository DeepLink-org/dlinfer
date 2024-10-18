#pragma once

#include <torch/custom_class.h>
#include <torch/script.h>

#include <memory>
#include <vector>

#include "model.h"

class DICPModel : public torch::CustomClassHolder {
public:
    DICPModel(const std::string& modelPath);
    ~DICPModel();
    void ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors, const std::string& param);

private:
    std::string modelPath_;
    std::shared_ptr<dicp::Model> model_;
    int modelId_ = 0;
    std::shared_ptr<atb::Context> context_;
};
