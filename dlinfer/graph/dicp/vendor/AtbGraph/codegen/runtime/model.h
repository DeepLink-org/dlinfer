#pragma once
#include <acl/acl.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <atb/types.h>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <map>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils/log.h"

namespace dicp {

template <typename T>
T getValue(const nlohmann::json& node, const std::string& key) {
    try {
        return node.at(key).get<T>();
    } catch (const std::exception& e) {
        DICP_LOG(ERROR) << "Error: " << e.what();
        DICP_LOG(ERROR) << "JSON Node: " << node.dump(4);
        throw std::runtime_error("getValue failed!");
    }
}

enum class TensorType {
    INTERMEDIATE_TENSOR = 0,
    NOT_INTERMEDIATE_TENSOR,
};

struct Node {
    std::shared_ptr<atb::Operation> operation;
    std::vector<atb::Tensor*> inTensors;
    std::vector<atb::Tensor*> outTensors;
    atb::VariantPack variantPack;
    atb::SVector<atb::ReshapeFunc> inTensorReshapeFuncs;
    atb::SVector<TensorType> inTensorTypes;
    atb::SVector<TensorType> outTensorTypes;
    std::unordered_map<int, int> inplaceIndices;
    uint64_t workspaceSize = 0;
    void* workspace = nullptr;
};

class Model;
struct Graph {
    friend class Model;
    std::vector<atb::Tensor> inTensors;
    std::vector<atb::Tensor> outTensors;
    std::vector<atb::Tensor> internalTensors;
    std::vector<Node> nodes;
    std::map<uint64_t, std::set<atb::Tensor*>> maxNodeIdTensorMap;
    void Init();
    std::string ToString() const;

private:
    void InitTensorType();
    bool IsInternalTensor(const atb::Tensor* tensor);
    void InitTensorMaxNodeMap();
};

class Model {
public:
    Model(const std::string& modelId, const std::string& modelPath);
    virtual ~Model();
    atb::Status Execute(atb::Context* context, std::vector<atb::Tensor>& inTensors, std::vector<atb::Tensor>& outTensors, const std::string& param);

private:
    int64_t BuildGraph();
    atb::Tensor CreateInternalTensorFromDesc(const atb::TensorDesc& tensorDesc);
    void CreateSingleOperation(const nlohmann::json& paramJson, Node& node);
    void CreateGraphOperation(const nlohmann::json& paramJson, Node& node);

    void BuildNodeVariantPack(int nodeId);
    atb::Status ExecuteNode(int nodeId);
    void ClearInternalTensors();
    atb::Tensor GetInternalTensor(atb::Tensor* outTensor, size_t nodeId, size_t outTensorId, const atb::TensorDesc& tensorDesc);
    void FreeInternalTensor(void* tensorDeviceData);
    void SetupInplaceOutputs(const nlohmann::json& inplaceOutputs, std::unordered_map<int, int>& inplaceIndices);
    void SetupReshapeFunctions(const nlohmann::json& reshapeInputs, atb::SVector<atb::ReshapeFunc>& funcs, size_t tensorSize);
    void SetupViewReshape(const nlohmann::json& reshapeInput, atb::ReshapeFunc& func);
    void SetupUnsqueezeReshape(const nlohmann::json& reshapeInput, atb::ReshapeFunc& func);
    void SetupSqueezeReshape(const nlohmann::json& reshapeInput, atb::ReshapeFunc& func);
    void SetupInferShape(const nlohmann::json& inferShape, atb::InferShapeFunc& inferShapeFunc);

private:
    std::string modelId_;
    std::string modelPath_;
    Graph graph_;
    atb::Context* context_;
    std::unordered_map<std::string, int> tensorsMap_;
    std::unordered_map<std::string, int> inputTensorsMap_;
    std::unordered_map<std::string, int> outputTensorsMap_;
    std::unordered_map<std::string, int> internalTensorsMap_;
    std::unordered_map<int32_t, std::unordered_map<int32_t, std::vector<int32_t>>> nodeHostTensorMap_;
    std::vector<std::pair<atb::Tensor, bool>> internalTensors_;
    std::vector<torch::Tensor> atInternalTensors_;
};
}  // namespace dicp
