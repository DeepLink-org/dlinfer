
file(GLOB_RECURSE ops_srcs CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(FILTER ops_srcs EXCLUDE REGEX "generated.*disabled$")
list(FILTER ops_srcs EXCLUDE REGEX "/op_api/")
list(FILTER ops_srcs EXCLUDE REGEX "dlinfer_grouped_matmul_direct_def.cpp$")
list(APPEND ops_srcs ${DLINFER_GMM_OPP_SOURCE_DIR}/common/tiling_util.cpp)

opbuild(OPS_SRC ${ops_srcs}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
        ENABLE_SOURCE ${ENABLE_SOURCE_PACAKGE}
)

if(ENABLE_CROSS_COMPILE)
    build_optiling_for_compile(
        OPS_SRC ${ops_srcs}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
    )
endif()

add_library(cust_optiling SHARED ${ops_srcs})
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_optiling PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_optiling PRIVATE
        intf_pub
        exe_graph
        register
        -Wl,--whole-archive
        tiling_api
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)

file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
file(GLOB aclnn_inc ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
add_library(cust_opapi SHARED
            ${ops_srcs}
            ${aclnn_src}
)
target_compile_definitions(cust_opapi PRIVATE ACLNN_WITH_BINARY OP_TILING_LIB)
include_directories(cust_opapi ${DLINFER_GMM_OPP_BINARY_DIR}/op_kernel/kernel/library)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_opapi PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_opapi PRIVATE intf_pub register ascend_kernels nnopbase ascendcl exe_graph tiling_api)
