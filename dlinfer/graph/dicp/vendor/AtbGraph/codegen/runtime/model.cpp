#include "model.h"

#include <atb/utils.h>

#include <algorithm>
#include <fstream>

#include "ops/operation_creator.h"
#include "utils/config.h"
#include "utils/log.h"
#include "utils/tensor_utils.h"
#include "utils/workspace.h"

namespace dicp {

static bool IsTensorDescEqual(const atb::TensorDesc& tensorDesc, const atb::Tensor& atbTensor) {
    if (atbTensor.desc.dtype != tensorDesc.dtype || atbTensor.desc.format != tensorDesc.format) {
        return false;
    }
    const auto& shape1 = atbTensor.desc.shape;
    const auto& shape2 = tensorDesc.shape;
    if (shape1.dimNum != shape2.dimNum) {
        return false;
    }
    return std::equal(shape1.dims, shape1.dims + shape1.dimNum, shape2.dims);
}

std::string Graph::ToString() const {
    std::stringstream ss;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << &inTensors.at(i) << " " << tensor_utils::TensorToString(inTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << &outTensors.at(i) << " " << tensor_utils::TensorToString(outTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < internalTensors.size(); ++i) {
        ss << "internalTensors[" << i << "]:" << &internalTensors.at(i) << " " << tensor_utils::TensorToString(internalTensors.at(i)) << std::endl;
    }
    ss << "nodes:" << nodes.size() << std::endl;

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes.at(i);
        ss << "node[" << i << "] operation:" << node.operation.get() << ", operationName:" << node.operation->GetName() << std::endl;
        for (auto tensorIt : node.inTensors) {
            ss << "node[" << i << "] inTensor:" << tensorIt << " " << tensor_utils::TensorToString(*tensorIt) << std::endl;
        }
        for (auto tensorIt : node.outTensors) {
            ss << "node[" << i << "] outTensor:" << tensorIt << " " << tensor_utils::TensorToString(*tensorIt) << std::endl;
        }
    }
    return ss.str();
}

void Graph::Init() {
    for (auto& node : nodes) {
        node.variantPack.inTensors.resize(node.inTensors.size());
        node.variantPack.outTensors.resize(node.outTensors.size());
    }
    InitTensorType();
    InitTensorMaxNodeMap();
}

void Graph::InitTensorType() {
    for (auto& node : nodes) {
        node.inTensorTypes.resize(node.inTensors.size());
        node.outTensorTypes.resize(node.outTensors.size());
        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            node.inTensorTypes.at(i) = IsInternalTensor(node.inTensors.at(i)) ? TensorType::INTERMEDIATE_TENSOR : TensorType::NOT_INTERMEDIATE_TENSOR;
        }
        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            node.outTensorTypes.at(i) = IsInternalTensor(node.outTensors.at(i)) ? TensorType::INTERMEDIATE_TENSOR : TensorType::NOT_INTERMEDIATE_TENSOR;
        }
    }
}

bool Graph::IsInternalTensor(const atb::Tensor* tensor) {
    return std::any_of(internalTensors.begin(), internalTensors.end(), [tensor](const atb::Tensor& internalTensor) { return &internalTensor == tensor; });
}

void Graph::InitTensorMaxNodeMap() {
    std::map<atb::Tensor*, uint64_t> tensorMaxNodeIdMap;
    maxNodeIdTensorMap.clear();

    for (size_t i = 0; i < internalTensors.size(); ++i) {
        atb::Tensor& internalTensor = internalTensors[i];
        uint64_t maxNodeId = 0;
        uint64_t dependNodeCount = 0;
        for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
            auto& node = nodes.at(nodeId);
            for (auto inTensorIt : node.inTensors) {
                if (&internalTensor == inTensorIt) {
                    maxNodeId = nodeId;
                    dependNodeCount++;
                }
            }
        }
        tensorMaxNodeIdMap[&internalTensor] = maxNodeId;
        DICP_LOG_IF(dependNodeCount == 0, INFO) << "runner graph internal tensor[" << i << "] dependNodeCount is 0.";
        maxNodeIdTensorMap[maxNodeId].insert(&internalTensor);
    }
}

atb::Tensor Model::CreateInternalTensorFromDesc(const atb::TensorDesc& tensorDesc) {
    torch::Tensor newAtTensor = tensor_utils::CreateAtTensorFromTensorDesc(tensorDesc);
    atInternalTensors_.push_back(newAtTensor);
    return tensor_utils::AtTensor2Tensor(newAtTensor);
}

Model::Model(const std::string& modelId, const std::string& modelPath) : modelId_(modelId), modelPath_(modelPath) {
    auto st = BuildGraph();
    DICP_LOG_IF(st != atb::NO_ERROR, ERROR) << modelId_ << " init graph:\n" << graph_.ToString();
    graph_.Init();
    DICP_LOG(INFO) << modelId_ << " init graph:\n" << graph_.ToString();
}

Model::~Model() {}

int64_t Model::BuildGraph() {
    // get json
    std::ifstream f(modelPath_);
    auto paramJson = nlohmann::json::parse(f);

    // parse json
    auto graphInputNames = getValue<std::vector<std::string>>(paramJson, "inputNames");
    auto graphOutputNames = getValue<std::vector<std::string>>(paramJson, "outputNames");
    auto graphInternalNames = getValue<std::vector<std::string>>(paramJson, "internalNames");

    int tensorCount = 0;
    graph_.inTensors.resize(graphInputNames.size());
    graph_.outTensors.resize(graphOutputNames.size());
    graph_.internalTensors.resize(graphInternalNames.size());
    for (unsigned int i = 0; i < graphInputNames.size(); ++i) {
        if (tensorsMap_.count(graphInputNames[i]) > 0) {
            DICP_LOG(ERROR) << "duplicate tensor name: " << graphInputNames[i];
            throw std::runtime_error("duplicate tensor name!");
        }
        tensorsMap_[graphInputNames[i]] = tensorCount++;
        inputTensorsMap_[graphInputNames[i]] = i;
    }
    for (unsigned int i = 0; i < graphOutputNames.size(); ++i) {
        if (tensorsMap_.count(graphOutputNames[i]) > 0) {
            DICP_LOG(ERROR) << "duplicate tensor name: " << graphOutputNames[i];
            throw std::runtime_error("duplicate tensor name");
        }
        tensorsMap_[graphOutputNames[i]] = tensorCount++;
        outputTensorsMap_[graphOutputNames[i]] = i;
    }
    for (unsigned int i = 0; i < graphInternalNames.size(); ++i) {
        if (tensorsMap_.count(graphInternalNames[i]) > 0) {
            DICP_LOG(ERROR) << "duplicate tensor name: " << graphInternalNames[i];
            throw std::runtime_error("duplicate tensor name");
        }
        tensorsMap_[graphInternalNames[i]] = tensorCount++;
        internalTensorsMap_[graphInternalNames[i]] = i;
    }

    for (const auto& node : paramJson["nodes"]) {
        auto nodeType = getValue<std::string>(node, "nodeType");
        auto nodeOp = node["value"];
        Node cur_node;

        if (nodeType == "singleOperation") {
            CreateSingleOperation(nodeOp, cur_node);
        } else if (nodeType == "graphOperation") {
            CreateGraphOperation(nodeOp, cur_node);
        } else {
            DICP_LOG(ERROR) << "invalid node type: " << nodeType;
            throw std::runtime_error("invalid node type!");
        }

        graph_.nodes.push_back(cur_node);
    }

    for (const auto& hostTensor : paramJson["hostTensorNames"]) {
        auto nodeId = getValue<int32_t>(hostTensor, "nodeId");
        auto tensorId = getValue<int32_t>(hostTensor, "tensorId");
        nodeHostTensorMap_[nodeId][tensorId] = {};
    }

    DICP_LOG(INFO) << "Model BuildGraph success";
    return atb::NO_ERROR;
}

atb::Status Model::Execute(atb::Context* context, std::vector<atb::Tensor>& inTensors, std::vector<atb::Tensor>& outTensors, const std::string& param) {
    if (graph_.inTensors.size() != inTensors.size() || graph_.outTensors.size() != outTensors.size()) {
        DICP_LOG(FATAL) << modelId_ << " graph.inTensors.size:" << graph_.inTensors.size() << ", inTensors.size:" << inTensors.size()
                        << ", graph.outTensors.size:" << graph_.outTensors.size() << ", outTensors.size:" << outTensors.size();
        return atb::ERROR_INVALID_GRAPH;
    }

    // get hostTensors
    nlohmann::json paramJson = nlohmann::json::parse(param);
    for (const auto& node : paramJson["hostTensors"]) {
        auto nodeId = getValue<int32_t>(node, "nodeId");
        auto tensorId = getValue<int32_t>(node, "tensorId");
        auto value = getValue<std::vector<int32_t>>(node, "value");
        nodeHostTensorMap_[nodeId][tensorId] = value;
    }

    ClearInternalTensors();
    context_ = context;
    graph_.inTensors = inTensors;
    graph_.outTensors = outTensors;
    DICP_LOG(INFO) << modelId_ << ", graph:\n" << graph_.ToString();

    for (size_t nodeId = 0; nodeId < graph_.nodes.size(); ++nodeId) {
        BuildNodeVariantPack(nodeId);
        atb::Status st = ExecuteNode(nodeId);
        DICP_LOG_IF(st != atb::NO_ERROR, FATAL) << modelId_ << " execute node[" << nodeId << "] failed, error code: " << st;
    }

    DICP_LOG(INFO) << modelId_ << " execute finshed!";
    return atb::NO_ERROR;
}

atb::Status Model::ExecuteNode(int nodeId) {
    auto& node = graph_.nodes.at(nodeId);
    atb::Status st = node.operation->Setup(node.variantPack, node.workspaceSize, context_);
    if (st != 0) {
        DICP_LOG(ERROR) << modelId_ << " setup node[" << nodeId << "] fail, not call execute";
        return st;
    }

    DICP_LOG(INFO) << modelId_ << " get node[" << nodeId << "] workspace size:" << node.workspaceSize;

    if (node.workspaceSize > 0) {
        node.workspace = GetWorkspaceBuffer(node.workspaceSize);
    }

    DICP_LOG(INFO) << modelId_ << "execute node[" << nodeId << "] start";

    st = node.operation->Execute(node.variantPack, (uint8_t*)(node.workspace), node.workspaceSize, context_);
    if (st != 0) {
        DICP_LOG(ERROR) << "execute node[" << nodeId << "] fail, error code: " << st;
    }
    return st;
}

void Model::BuildNodeVariantPack(int nodeId) {
    auto& node = graph_.nodes.at(nodeId);
    bool needReshape = node.inTensorReshapeFuncs.size() > 0;

    atb::SVector<atb::TensorDesc> inTensorDescs;
    inTensorDescs.resize(node.variantPack.inTensors.size());
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
        inTensorDescs.at(i) = node.inTensors.at(i)->desc;
        if (needReshape) {
            node.inTensorReshapeFuncs.at(i)(node.inTensors.at(i)->desc.shape, inTensorDescs.at(i).shape);
            node.variantPack.inTensors.at(i).desc.shape = inTensorDescs.at(i).shape;
        }
        DICP_LOG(INFO) << modelId_ << " nodes[" << nodeId << "] inTensors[" << i << "]:" << tensor_utils::TensorToString(node.variantPack.inTensors.at(i));
    }

    atb::SVector<atb::TensorDesc> outTensorDescs;
    outTensorDescs.resize(node.operation->GetOutputNum());

    atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);
    DICP_LOG_IF(st != 0, FATAL) << modelId_ << " nodes[" << nodeId << "] "
                                << " infer shape fail, error code: " << st;

    bool hasInplaceOutputs = node.inplaceIndices.size() > 0;
    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        if (hasInplaceOutputs && node.inplaceIndices.count(i) > 0) {
            auto inputIdx = node.inplaceIndices[i];
            node.variantPack.outTensors.at(i) = node.variantPack.inTensors.at(inputIdx);
            *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
            continue;
        }

        node.variantPack.outTensors.at(i) = *node.outTensors.at(i);
        if (node.outTensorTypes.at(i) == TensorType::INTERMEDIATE_TENSOR) {
            node.variantPack.outTensors.at(i) = GetInternalTensor(node.outTensors.at(i), nodeId, i, outTensorDescs.at(i));
            *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
        }
    }

    auto it = graph_.maxNodeIdTensorMap.find(nodeId);
    if (it != graph_.maxNodeIdTensorMap.end()) {
        for (auto tensorIt : it->second) {
            FreeInternalTensor(tensorIt->deviceData);
        }
    }

    // bind host inTensors
    if (nodeHostTensorMap_.count(nodeId) > 0) {
        for (auto& i : nodeHostTensorMap_[nodeId]) {
            node.variantPack.inTensors.at(i.first).hostData = i.second.data();
        }
        DICP_LOG(INFO) << "create host inputs end.";
    }
}

void Model::CreateSingleOperation(const nlohmann::json& paramJson, Node& node) {
    auto opType = getValue<std::string>(paramJson, "type");
    auto opName = getValue<std::string>(paramJson, "name");
    auto opInputNames = getValue<std::vector<std::string>>(paramJson, "inputNames");
    auto opOutputNames = getValue<std::vector<std::string>>(paramJson, "outputNames");
    atb::Operation* op = CreateOperation(opType, paramJson["param"]);

    node.operation.reset(op);
    for (const auto& t : opInputNames) {
        if (inputTensorsMap_.count(t) > 0) {
            node.inTensors.push_back(&graph_.inTensors[inputTensorsMap_[t]]);
        } else if (internalTensorsMap_.count(t) > 0) {
            node.inTensors.push_back(&graph_.internalTensors[internalTensorsMap_[t]]);
        } else if (outputTensorsMap_.count(t) > 0) {
            node.inTensors.push_back(&graph_.outTensors[outputTensorsMap_[t]]);
        } else {
            DICP_LOG(ERROR) << "cannot find name in input/internal: " << t;
            throw std::runtime_error("cannot find name in input/internal!");
        }
    }
    for (const auto& t : opOutputNames) {
        if (outputTensorsMap_.count(t) > 0) {
            node.outTensors.push_back(&graph_.outTensors[outputTensorsMap_[t]]);
        } else if (internalTensorsMap_.count(t) > 0) {
            node.outTensors.push_back(&graph_.internalTensors[internalTensorsMap_[t]]);
        } else {
            DICP_LOG(ERROR) << "cannot find name in output/internal: " << t;
            throw std::runtime_error("cannot find name in input/internal!");
        }
    }
    if (getValue<bool>(paramJson, "hasReshapeInputs")) {
        SetupReshapeFunctions(paramJson["reshapeInputs"], node.inTensorReshapeFuncs, node.inTensors.size());
    }
    if (getValue<bool>(paramJson, "hasInplaceOutputs")) {
        SetupInplaceOutputs(paramJson["inplaceOutputs"], node.inplaceIndices);
    }
}

void Model::CreateGraphOperation(const nlohmann::json& paramJson, Node& node) {
    atb::GraphParam graph_param;
    int nodeSize = getValue<int32_t>(paramJson, "nodeSize");
    auto inputNames = getValue<std::vector<std::string>>(paramJson, "inputNames");
    auto outputNames = getValue<std::vector<std::string>>(paramJson, "outputNames");
    auto internalNames = getValue<std::vector<std::string>>(paramJson, "internalNames");
    graph_param.inTensorNum = inputNames.size();
    graph_param.outTensorNum = outputNames.size();
    graph_param.internalTensorNum = internalNames.size();
    graph_param.nodes.resize(nodeSize);

    // graph local tensor ids
    std::unordered_map<std::string, int> graph_tensor_ids;
    int tensorCount = 0;
    for (unsigned int i = 0; i < inputNames.size(); ++i) {
        graph_tensor_ids[inputNames[i]] = tensorCount++;
    }
    for (unsigned int i = 0; i < outputNames.size(); ++i) {
        graph_tensor_ids[outputNames[i]] = tensorCount++;
    }
    for (unsigned int i = 0; i < internalNames.size(); ++i) {
        graph_tensor_ids[internalNames[i]] = tensorCount++;
    }

    int cur_node_index = 0;
    for (const auto& node : paramJson["nodes"]) {
        auto nodeType = getValue<std::string>(node, "nodeType");
        auto nodeOp = node["value"];
        if (nodeType == "singleOperation") {
            auto opType = getValue<std::string>(nodeOp, "type");
            auto opName = getValue<std::string>(nodeOp, "name");
            auto opInputNames = getValue<std::vector<std::string>>(nodeOp, "inputNames");
            auto opOutputNames = getValue<std::vector<std::string>>(nodeOp, "outputNames");
            atb::Operation* op = CreateOperation(opType, nodeOp["param"]);
            graph_param.nodes[cur_node_index].operation = op;
            for (const auto& t : opInputNames) {
                graph_param.nodes[cur_node_index].inTensorIds.push_back(graph_tensor_ids[t]);
            }
            for (const auto& t : opOutputNames) {
                graph_param.nodes[cur_node_index].outTensorIds.push_back(graph_tensor_ids[t]);
            }

            if (getValue<bool>(nodeOp, "hasReshapeInputs")) {
                auto& cur_node = graph_param.nodes[cur_node_index];
                SetupReshapeFunctions(nodeOp["reshapeInputs"], cur_node.inTensorReshapeFuncs, cur_node.inTensorIds.size());
            }
        } else {
            DICP_LOG(ERROR) << "invalid node type in graph opearation, ndoeType: " << nodeType;
            throw std::runtime_error("invalid node type in graph opearation!");
        }
        cur_node_index++;
    }
    if (getValue<bool>(paramJson, "hasInferShape")) {
        SetupInferShape(paramJson["inferShape"], graph_param.inferShapeFunc);
    }
    if (getValue<bool>(paramJson, "hasInplaceOutputs")) {
        SetupInplaceOutputs(paramJson["inplaceOutputs"], node.inplaceIndices);
    }

    atb::Operation* op = nullptr;
    auto st = atb::CreateOperation(graph_param, &op);
    if (st != 0) {
        DICP_LOG(ERROR) << "atb CreateOperation graph failed, st: " << st;
        throw std::runtime_error("atb CreateOperation graph failed!");
    }

    // bind Model tensor ids to graph tensor
    node.operation.reset(op);
    for (const auto& t : inputNames) {
        if (inputTensorsMap_.count(t) > 0) {
            node.inTensors.push_back(&graph_.inTensors[inputTensorsMap_[t]]);
        } else {
            node.inTensors.push_back(&graph_.internalTensors[internalTensorsMap_[t]]);
        }
    }
    for (const auto& t : outputNames) {
        if (outputTensorsMap_.count(t) > 0) {
            node.outTensors.push_back(&graph_.outTensors[outputTensorsMap_[t]]);
        } else {
            node.outTensors.push_back(&graph_.internalTensors[internalTensorsMap_[t]]);
        }
    }
}

void Model::SetupInferShape(const nlohmann::json& inferShape, atb::InferShapeFunc& inferShapeFunc) {
    auto inferType = getValue<std::string>(inferShape, "type");
    if (inferType == "equal") {
        auto outputByInput = getValue<std::vector<int32_t>>(inferShape, "value");
        inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) {
            for (size_t i = 0; i < outTensorDescs.size(); ++i) {
                outTensorDescs.at(i) = inTensorDescs.at(outputByInput[i]);
            }
            return atb::NO_ERROR;
        };
    }
}

void Model::SetupInplaceOutputs(const nlohmann::json& inplaceOutputs, std::unordered_map<int, int>& inplaceIndices) {
    for (const auto& inplaceTensors : inplaceOutputs) {
        auto outputIdx = getValue<int32_t>(inplaceTensors, "output_index");
        auto inputIdx = getValue<int32_t>(inplaceTensors, "input_index");
        inplaceIndices[outputIdx] = inputIdx;
    }
}

void Model::SetupReshapeFunctions(const nlohmann::json& reshapeInputs, atb::SVector<atb::ReshapeFunc>& funcs, size_t tensorSize) {
    funcs.resize(tensorSize);
    int count = 0;
    for (const auto& reshapeInput : reshapeInputs) {
        auto reshapeType = getValue<std::string>(reshapeInput, "reshapeType");
        if (reshapeType == "None") {
            funcs.at(count) = [](const atb::Dims& oldShape, atb::Dims& newShape) { newShape = oldShape; };
        } else if (reshapeType == "view") {
            SetupViewReshape(reshapeInput, funcs.at(count));
        } else if (reshapeType == "unsqueeze") {
            SetupUnsqueezeReshape(reshapeInput, funcs.at(count));
        } else if (reshapeType == "squeeze") {
            SetupSqueezeReshape(reshapeInput, funcs.at(count));
        }
        count++;
    }
}

void Model::SetupViewReshape(const nlohmann::json& reshapeInput, atb::ReshapeFunc& func) {
    auto dimNum = getValue<int32_t>(reshapeInput, "dimNum");
    auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dims");
    bool needInferDim = false;
    size_t dimNeedInfer = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == -1) {
            needInferDim = true;
            dimNeedInfer = i;
            break;
        }
    }
    func = [=](const atb::Dims& oldShape, atb::Dims& newShape) {
        newShape.dimNum = dimNum;
        if (needInferDim) {
            int64_t totalValue = 1;
            int64_t otherProd = 1;
            for (size_t i = 0; i < oldShape.dimNum; ++i) {
                totalValue *= oldShape.dims[i];
            }
            for (size_t i = 0; i < dims.size(); ++i) {
                if (i != dimNeedInfer) {
                    otherProd *= dims[i];
                }
            }
            newShape.dims[dimNeedInfer] = totalValue / otherProd;
        }
        for (size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] != -1) {
                newShape.dims[i] = dims[i];
            }
        }
    };
}

void Model::SetupUnsqueezeReshape(const nlohmann::json& reshapeInput, atb::ReshapeFunc& func) {
    auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dim");
    func = [=](const atb::Dims& oldShape, atb::Dims& newShape) {
        std::vector<int64_t> dimValues(oldShape.dims, oldShape.dims + oldShape.dimNum);
        for (const auto& d : dims) {
            int offset = d < 0 ? d + oldShape.dimNum + 1 : d;
            dimValues.insert(dimValues.begin() + offset, 1);
        }
        newShape.dimNum = dimValues.size();
        std::copy(dimValues.begin(), dimValues.end(), newShape.dims);
    };
}

void Model::SetupSqueezeReshape(const nlohmann::json& reshapeInput, atb::ReshapeFunc& func) {
    auto dims = getValue<std::vector<int32_t>>(reshapeInput, "dim");
    func = [=](const atb::Dims& oldShape, atb::Dims& newShape) {
        std::vector<int64_t> dimValues(oldShape.dims, oldShape.dims + oldShape.dimNum);
        for (const auto& d : dims) {
            int offset = d < 0 ? d + oldShape.dimNum : d;
            dimValues.erase(dimValues.begin() + offset);
        }
        newShape.dimNum = dimValues.size();
        std::copy(dimValues.begin(), dimValues.end(), newShape.dims);
    };
}

void Model::ClearInternalTensors() {
    internalTensors_.clear();
    atInternalTensors_.clear();
}

atb::Tensor Model::GetInternalTensor(atb::Tensor* outTensor, size_t nodeId, size_t outTensorId, const atb::TensorDesc& tensorDesc) {
    auto it = std::find_if(internalTensors_.begin(), internalTensors_.end(), [&tensorDesc](std::pair<atb::Tensor, bool>& tensorPair) {
        return !tensorPair.second && IsTensorDescEqual(tensorDesc, tensorPair.first);
    });

    if (it != internalTensors_.end()) {
        it->second = true;
        DICP_LOG(INFO) << modelId_ << " use old internal tensor";
        return it->first;
    }

    DICP_LOG(INFO) << modelId_ << " create internal tensor, node[" << nodeId << "], outTensor[" << outTensorId << "]";
    atb::Tensor newTensor = CreateInternalTensorFromDesc(tensorDesc);
    internalTensors_.emplace_back(newTensor, true);
    return newTensor;
}

void Model::FreeInternalTensor(void* tensorDeviceData) {
    auto it = std::find_if(internalTensors_.begin(), internalTensors_.end(), [tensorDeviceData](const std::pair<atb::Tensor, bool>& tensorPair) {
        return tensorPair.first.deviceData == tensorDeviceData;
    });

    if (it != internalTensors_.end()) {
        it->second = false;
        DICP_LOG(INFO) << modelId_ << " free internal tensor";
    }
}

}  // namespace dicp
