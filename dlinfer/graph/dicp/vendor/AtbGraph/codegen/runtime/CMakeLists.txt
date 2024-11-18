cmake_minimum_required(VERSION 3.18)
project(dicp_model LANGUAGES CXX)

include(ascend)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

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

target_include_directories(
    dicp_model PUBLIC
    ${THIRD_PARTY_DIR}/json/single_include
    ${THIRD_PARTY_DIR}/spdlog/include
    ${THIRD_PARTY_DIR}/half/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}/aclnn
    ${ATB_INCLUDE_DIRS}
)

target_compile_options(dicp_model PRIVATE ${COMPILE_OPTIONS})

target_link_libraries(dicp_model PUBLIC
    Python::Python
    torch
    ${TORCH_NPU_LIBRARY}
    ${CANN_LIBRARY}
    ${ATB_LIBRARY}
)

file(RELATIVE_PATH OUTPUT_LIB_RELATIVE_PATH "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../")
install(
    TARGETS dicp_model
    DESTINATION ${OUTPUT_LIB_RELATIVE_PATH}
)
