cmake_minimum_required(VERSION 3.18)
project(ascend_extension)

execute_process(
    COMMAND python -c "from torch.utils import cmake_prefix_path; \
    print(cmake_prefix_path + '/Torch', end='')"
    OUTPUT_VARIABLE Torch_DIR
)

execute_process(
    COMMAND bash -c "pip show torch-npu | awk '/^Location:/{ORS=\"\"; print \$2\"/torch_npu\"}'"
    OUTPUT_VARIABLE Torch_npu_ROOT
)

execute_process(
    COMMAND python -c "import torch; \
    print('1' if torch.compiled_with_cxx11_abi() else '0', end='')"
    OUTPUT_VARIABLE _GLIBCXX_USE_CXX11_ABI
)

find_package(Torch REQUIRED)
find_package(Torch_npu REQUIRED)
find_package(Hccl REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'")

set(CSRC_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe_gating_topk_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/op_api_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/torch_npu_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quant_matmul.cpp
)

add_library(
    ${PROJECT_NAME} SHARED
    ${CSRC_FILES}
)

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    PREFIX ""
)

target_compile_definitions(
    ${PROJECT_NAME} PUBLIC
    GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${TORCH_NPU_INCLUDE_DIRS}
    PUBLIC ${HCCL_INCLUDE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC Python::Python
    PUBLIC torch
    PUBLIC ${TORCH_NPU_LIBRARY}
    PUBLIC ${HCCL_LIBRARY}
)

install(
    TARGETS ${PROJECT_NAME}
    DESTINATION ./dlinfer/vendor/ascend
)
