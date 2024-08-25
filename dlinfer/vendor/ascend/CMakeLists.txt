cmake_minimum_required(VERSION 3.20)
project(ascend_extension)

find_package(Torch_npu REQUIRED)
find_package(Hccl REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'")

set(CSRC_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe_gating_topk_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/op_api_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/torch_npu_utils.cpp
)

add_library(
    ${PROJECT_NAME} SHARED
    ${CSRC_FILES}
)

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    PREFIX ""
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
