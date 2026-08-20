// Copied from DeepLink-org/DLBlas csrc/ascend/grouped_matmul_direct and
// adapted for DLInfer pybind packaging.
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_dlinfer_grouped_matmul_direct.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

constexpr int64_t kDirectGroups = 2560;

at::Tensor GroupedMatmulDirect(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& group_list, int64_t group_list_type, uint64_t stream_handle);

aclDataType ToAclDataType(const at::ScalarType dtype) {
    switch (dtype) {
        case at::ScalarType::Half:
            return ACL_FLOAT16;
        case at::ScalarType::BFloat16:
            return ACL_BF16;
        case at::ScalarType::Long:
            return ACL_INT64;
        default:
            TORCH_CHECK(false, "unsupported dtype for direct grouped matmul: ", dtype);
    }
}

struct AclTensorDeleter {
    void operator()(aclTensor* tensor) const {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }
};

struct AclTensorListDeleter {
    void operator()(aclTensorList* tensors) const {
        if (tensors != nullptr) {
            aclDestroyTensorList(tensors);
        }
    }
};

using AclTensorPtr = std::unique_ptr<aclTensor, AclTensorDeleter>;
using AclTensorListPtr = std::unique_ptr<aclTensorList, AclTensorListDeleter>;

AclTensorPtr MakeAclTensor(const at::Tensor& tensor) {
    std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
    std::vector<int64_t> strides(tensor.strides().begin(), tensor.strides().end());
    auto* acl_tensor = aclCreateTensor(shape.data(),
                                       shape.size(),
                                       ToAclDataType(tensor.scalar_type()),
                                       strides.data(),
                                       tensor.storage_offset(),
                                       ACL_FORMAT_ND,
                                       shape.data(),
                                       shape.size(),
                                       tensor.data_ptr());
    TORCH_CHECK(acl_tensor != nullptr, "aclCreateTensor failed");
    return AclTensorPtr(acl_tensor);
}

AclTensorPtr MakeTransposedWeightAclTensor(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.dim() == 3 && tensor.is_contiguous(), "stored weight must be contiguous [E, N, K]");
    const std::array<int64_t, 3> view_shape{tensor.size(0), tensor.size(2), tensor.size(1)};
    // The kernel consumes the raw [E, N, K] storage with transposeWeight=true,
    // while host tiling needs the logical [E, K, N] shape. Mark the ACL view
    // contiguous so the generic individual-op executor does not insert a copy.
    const std::array<int64_t, 3> view_strides{
        view_shape[1] * view_shape[2], view_shape[2], 1};
    const std::array<int64_t, 3> storage_shape{tensor.size(0), tensor.size(1), tensor.size(2)};
    auto* acl_tensor = aclCreateTensor(view_shape.data(),
                                       view_shape.size(),
                                       ToAclDataType(tensor.scalar_type()),
                                       view_strides.data(),
                                       tensor.storage_offset(),
                                       ACL_FORMAT_ND,
                                       storage_shape.data(),
                                       storage_shape.size(),
                                       tensor.data_ptr());
    TORCH_CHECK(acl_tensor != nullptr, "aclCreateTensor failed for weight");
    return AclTensorPtr(acl_tensor);
}

AclTensorListPtr MakeAclTensorList(aclTensor* tensor) {
    std::array<aclTensor*, 1> tensors{tensor};
    auto* list = aclCreateTensorList(tensors.data(), tensors.size());
    TORCH_CHECK(list != nullptr, "aclCreateTensorList failed");
    return AclTensorListPtr(list);
}

AclTensorPtr MakeEmptyAclTensor(const aclDataType dtype) {
    constexpr std::array<int64_t, 1> shape{0};
    constexpr std::array<int64_t, 1> strides{1};
    auto* tensor = aclCreateTensor(shape.data(),
                                   shape.size(),
                                   dtype,
                                   strides.data(),
                                   0,
                                   ACL_FORMAT_ND,
                                   shape.data(),
                                   shape.size(),
                                   nullptr);
    TORCH_CHECK(tensor != nullptr, "aclCreateTensor failed for empty input");
    return AclTensorPtr(tensor);
}

AclTensorListPtr MakeEmptyAclTensorList(const aclDataType dtype) {
    auto tensor = MakeEmptyAclTensor(dtype);
    auto* raw_tensor = tensor.get();
    auto* list = aclCreateTensorList(&raw_tensor, 1);
    TORCH_CHECK(list != nullptr, "aclCreateTensorList failed for empty input");
    tensor.release();
    return AclTensorListPtr(list);
}

at::Tensor GroupedMatmulDirect(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& group_list, int64_t group_list_type, uint64_t stream_handle) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be an NPU tensor");
    TORCH_CHECK(weight.device() == x.device() && group_list.device() == x.device(), "x, weight and group_list must be on the same NPU device");
    TORCH_CHECK(x.dim() == 2, "x must have shape [M, K]");
    TORCH_CHECK(weight.dim() == 3 && weight.size(0) == kDirectGroups, "weight must have stored shape [2560, N, K]");
    TORCH_CHECK(group_list.dim() == 1 && group_list.numel() == kDirectGroups, "group_list must contain 2560 entries");
    TORCH_CHECK(group_list.scalar_type() == at::ScalarType::Long, "group_list must be int64");
    TORCH_CHECK(group_list_type == 1, "direct2560 only supports group_list_type=1");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16, "direct2560 only supports float16 and bfloat16");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "x and weight dtypes must match");
    TORCH_CHECK(x.size(1) == weight.size(2), "x K must match weight K");

    auto out = at::empty({x.size(0), weight.size(1)}, x.options());
    auto workspace = at::Tensor();

    auto x_acl = MakeAclTensor(x);
    auto weight_acl = MakeTransposedWeightAclTensor(weight);
    auto group_list_acl = MakeAclTensor(group_list);
    auto out_acl = MakeAclTensor(out);
    auto x_list = MakeAclTensorList(x_acl.get());
    auto weight_list = MakeAclTensorList(weight_acl.get());
    auto out_list = MakeAclTensorList(out_acl.get());
    auto empty_bias = MakeEmptyAclTensorList(ToAclDataType(x.scalar_type()));
    auto empty_scale = MakeEmptyAclTensorList(ACL_UINT64);
    auto empty_offset = MakeEmptyAclTensorList(ACL_FLOAT);
    auto empty_antiquant_scale = MakeEmptyAclTensorList(ToAclDataType(x.scalar_type()));
    auto empty_antiquant_offset = MakeEmptyAclTensorList(ToAclDataType(x.scalar_type()));
    auto empty_per_token_scale = MakeEmptyAclTensor(ACL_FLOAT);
    // aclTensorList owns the tensors passed to aclCreateTensorList. Transfer
    // ownership to the lists so the individual guards do not destroy them a
    // second time when this function returns.
    x_acl.release();
    weight_acl.release();
    out_acl.release();

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnDlinferGroupedMatmulDirectGetWorkspaceSize(
        x_list.get(),
        weight_list.get(),
        empty_bias.get(),
        empty_scale.get(),
        empty_offset.get(),
        empty_antiquant_scale.get(),
        empty_antiquant_offset.get(),
        group_list_acl.get(),
        empty_per_token_scale.get(),
        2,
        0,
        true,
        false,
        0,
        group_list_type,
        0,
        nullptr,
        out_list.get(),
        &workspace_size,
        &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "aclnnDlinferGroupedMatmulDirectGetWorkspaceSize failed, status=", status);
    TORCH_CHECK(executor != nullptr, "GroupedMatmul returned a null executor");

    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        workspace = at::empty({static_cast<int64_t>(workspace_size)}, x.options().dtype(at::ScalarType::Byte));
        workspace_addr = workspace.data_ptr();
    }

    TORCH_CHECK(stream_handle != 0, "current NPU stream handle must not be null");
    const auto stream = reinterpret_cast<aclrtStream>(stream_handle);
    const auto execute_status = aclnnDlinferGroupedMatmulDirect(workspace_addr, workspace_size, executor, stream);
    TORCH_CHECK(execute_status == ACL_SUCCESS, "aclnnDlinferGroupedMatmulDirect failed, status=", execute_status);
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("grouped_matmul", &GroupedMatmulDirect, "DLInfer bundled GroupedMatmul direct2560"); }
