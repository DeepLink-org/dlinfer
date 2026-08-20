
function(get_system_info SYSTEM_INFO)
  if (UNIX)
    execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO} ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)
  elseif (WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else ()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif ()
endfunction()

function(opbuild)
  message(STATUS "Opbuild generating sources")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX;ENABLE_SOURCE" "OPS_SRC" ${ARGN})
  execute_process(COMMAND ${CMAKE_COMPILE} -g -fPIC -shared -std=c++17 ${OPBUILD_OPS_SRC} -D_GLIBCXX_USE_CXX11_ABI=0
                  -I ${ASCEND_CANN_PACKAGE_PATH}/include
                  -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
                  -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnn
                  -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/base
                  -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
                  -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common
                  -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/base
                  -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/tiling
                  -I ${DLINFER_GMM_OPP_SOURCE_DIR}/common/include
                  -I ${DLINFER_GMM_OPP_SOURCE_DIR}/op_host
                  -I ${DLINFER_GMM_OPP_SOURCE_DIR}/op_host/op_tiling
                  -I ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
                  -L ${ASCEND_CANN_PACKAGE_PATH}/lib64 -lexe_graph -lregister -ltiling_api
                  -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if (NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if (NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()

  set(ENV{OPS_PRODUCT_NAME} ${ASCEND_COMPUTE_UNIT})
  set(ENV{ENABLE_SOURCE_PACAKGE} ${OPBUILD_ENABLE_SOURCE})
  execute_process(COMMAND ${proj_env} ${prefix_env} ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
                          ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  unset(ENV{OPS_PRODUCT_NAME})
  unset(ENV{ENABLE_SOURCE_PACAKGE})
  if (${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message(FATAL_ERROR "opbuild ops error: ${EXEC_ERROR}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

function(build_optiling_for_compile)
  message(STATUS "building optiling so for compile")
  cmake_parse_arguments(TILING_COMPILE "" "OUT_DIR" "OPS_SRC" ${ARGN})
  file(MAKE_DIRECTORY ${TILING_COMPILE_OUT_DIR}/op_impl/ai_core/tbe/op_tiling/)
  execute_process(COMMAND ${CMAKE_COMPILE} -fPIC -shared -std=c++11 ${TILING_COMPILE_OPS_SRC} -D_GLIBCXX_USE_CXX11_ABI=0
                  -I ${ASCEND_CANN_PACKAGE_PATH}/include -I ${CMAKE_CURRENT_SOURCE_DIR}/../op_host -L ${ASCEND_CANN_PACKAGE_PATH}/lib64
                  -DOP_TILING_LIB -fvisibility=hidden -lexe_graph -lregister -Wl, --whole-archive -ltiling_api -lrt2_registry -Wl, --no-whole-archive
                  -o ${TILING_COMPILE_OUT_DIR}/op_impl/ai_core/tbe/op_tiling/liboptiling.so
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("build optiling lib for compile info: ${EXEC_INFO}")
    message("build optiling lib for compile error: ${EXEC_ERROR}")
    message(FATAL_ERROR "optiling lib for compile failed")
  endif()
  message(STATUS "building optiling so for compile - done")
endfunction()

function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${DLINFER_GMM_OPP_SOURCE_DIR}/cmake/util/ascendc_gen_options.py
                          ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS} ${OP_TYPE} ${OP_COMPILE_COMPUTE_UNIT}
                          ${OP_COMPILE_OPTIONS}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR)
  if (${EXEC_RESULT})
      message("add ops compile options info: ${EXEC_INFO}")
      message("add ops compile options error: ${EXEC_ERROR}")
      message(FATAL_ERROR "add ops compile options failed!")
  endif()
endfunction()

function(add_kernel_compile op_type src)
  cmake_parse_arguments(BINCMP "" "OPS_INFO;OUT_DIR;TILING_LIB" "COMPUTE_UNIT;OPTIONS;CONFIGS" ${ARGN})
  if (NOT DEFINED BINCMP_COMPUTE_UNIT)
    set(BINCMP_COMPUTE_UNIT ${ASCEND_COMPUTE_UNIT})
  endif()
  if (NOT DEFINED BINCMP_OUT_DIR)
    set(BINCMP_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()
  set(BINCMP_OUT_DIR ${BINCMP_OUT_DIR}/kernel)
  if (NOT DEFINED BINCMP_TILING_LIB)
    set(BINCMP_TILING_LIB $<TARGET_FILE:cust_optiling>)
  endif()
  if (NOT TARGET op_kernel_pack)
    add_custom_target(op_kernel_pack
                      COMMAND ${CMAKE_COMMAND} -E env
                      "PYTHONDONTWRITEBYTECODE=1"
                      "PYTHONPATH=${ASCEND_COMPILER_PYTHONPATH}"
                      ${ASCEND_PYTHON_EXECUTABLE} -S
                      ${DLINFER_GMM_OPP_SOURCE_DIR}/cmake/util/ascendc_pack_kernel.py
                      --input-path=${BINCMP_OUT_DIR}
                      --output-path=${BINCMP_OUT_DIR}/library)
    add_library(ascend_kernels INTERFACE)
    target_link_libraries(ascend_kernels INTERFACE kernels)
    target_link_directories(ascend_kernels INTERFACE ${BINCMP_OUT_DIR}/library)
    target_include_directories(ascend_kernels INTERFACE ${BINCMP_OUT_DIR}/library)
    add_dependencies(ascend_kernels op_kernel_pack)
  endif()

  # add Environment Variable Configurations of ccache
  set(_ASCENDC_ENV_VAR)
  if(${CMAKE_CXX_COMPILER_LAUNCHER} MATCHES "ccache$")
    list(APPEND _ASCENDC_ENV_VAR export ASCENDC_CCACHE_EXECUTABLE=${CMAKE_CXX_COMPILER_LAUNCHER} &&)
  endif()

  foreach(compute_unit ${BINCMP_COMPUTE_UNIT})
    if (NOT DEFINED BINCMP_OPS_INFO)
      set(BINCMP_OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini)
    endif()
    add_custom_target(${op_type}_${compute_unit}
                     COMMAND ${_ASCENDC_ENV_VAR} ${CMAKE_COMMAND} -E env
                     "PYTHONDONTWRITEBYTECODE=1"
                     ${ASCEND_PYTHON_EXECUTABLE} ${DLINFER_GMM_OPP_SOURCE_DIR}/cmake/util/ascendc_compile_kernel.py
                     --op-name=${op_type}
                     --src-file=${src}
                     --compute-unit=${compute_unit}
                     --compile-options=\"${BINCMP_OPTIONS}\"
                     --debug-config=\"${BINCMP_CONFIGS}\"
                     --config-ini=${BINCMP_OPS_INFO}
                     --tiling-lib=${BINCMP_TILING_LIB}
                     --output-path=${BINCMP_OUT_DIR})
    add_dependencies(${op_type}_${compute_unit} cust_optiling)
    add_dependencies(op_kernel_pack ${op_type}_${compute_unit})
  endforeach()
endfunction()

function(add_cross_compile_target)
    cmake_parse_arguments(CROSSMP "" "TARGET;OUT_DIR;INSTALL_DIR" "" ${ARGN})
    add_custom_target(${CROSSMP_TARGET} ALL
                      DEPENDS ${CROSSMP_OUT_DIR}
    )
    install(DIRECTORY ${CROSSMP_OUT_DIR}
            DESTINATION ${CROSSMP_INSTALL_DIR}
    )
endfunction()
