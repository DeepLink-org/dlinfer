#include "op_api_common.hpp"

void* GetOpApiFuncAddrFromFeatureLib(const char* api_name) {
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_infer_handler, "libaclnn_ops_infer.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_train_handler, "libaclnn_ops_train.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(adv_infer_handler, "libaclnn_adv_infer.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(adv_train_handler, "libaclnn_adv_train.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(dvpp_handler, "libacl_dvpp_op.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(sparse_handler, "libaclsparse.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(optim_handler, "libacloptim.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(fft_handler, "libaclfft.so");
    GET_OP_API_FUNC_FROM_FEATURE_LIB(rand_handler, "libaclrand.so");
    return nullptr;
}
