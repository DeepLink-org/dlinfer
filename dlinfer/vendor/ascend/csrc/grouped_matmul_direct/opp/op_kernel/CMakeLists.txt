
# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0 --cce-ignore-always-inline=true)
endif()

# Multi-operator should add OpName and kernel file
add_kernel_compile(DlinferGroupedMatmulDirect ${CMAKE_CURRENT_SOURCE_DIR}/dlinfer_grouped_matmul_direct.cpp)

if (ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
endif()
