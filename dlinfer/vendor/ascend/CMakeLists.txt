cmake_minimum_required(VERSION 3.18)
project(ascend_extension)

include(ascend)

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

target_compile_definitions(
    ${PROJECT_NAME} PUBLIC
    GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME} PRIVATE
    Python::Python
    torch
    ${TORCH_NPU_LIBRARY}
    ${CANN_LIBRARY}
)

file(RELATIVE_PATH OUTPUT_LIB_RELATIVE_PATH "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
install(
    TARGETS ${PROJECT_NAME}
    DESTINATION ${OUTPUT_LIB_RELATIVE_PATH}
)
