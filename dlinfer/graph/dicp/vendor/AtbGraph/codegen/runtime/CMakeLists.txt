cmake_minimum_required(VERSION 3.18)
project(dlinfer LANGUAGES CXX)

execute_process(
    COMMAND python3 -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND python3 -c "import torch_npu; print(torch_npu.__path__[0])"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Torch path: ${TORCH_PATH}")
message(STATUS "Torch NPU path: ${TORCH_NPU_PATH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

include_directories(
    $ENV{PYTHON_INCLUDE_PATH}
    ${THIRD_PARTY_DIR}/json/single_include
    ${THIRD_PARTY_DIR}/spdlog/include
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
    ${TORCH_NPU_PATH}/include
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/aclnn_ops
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

link_directories(
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_TOOLKIT_HOME}/lib64
    ${TORCH_PATH}/lib
    ${TORCH_NPU_PATH}/lib
)

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/aclnn_ops/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
)

set(COMPILE_OPTIONS
    -Wno-unused-function
    -Wno-unused-variable
    -Wno-unused-parameter
    -Wno-attributes
    -D_GLIBCXX_USE_CXX11_ABI=0
)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

add_library(dicp_model SHARED ${SOURCES})
set_target_properties(dicp_model PROPERTIES
    OUTPUT_NAME "dicp_model"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_compile_options(dicp_model PRIVATE ${COMPILE_OPTIONS})

target_link_libraries(dicp_model PRIVATE
    atb torch c10 torch_cpu torch_python torch_npu opapi)
