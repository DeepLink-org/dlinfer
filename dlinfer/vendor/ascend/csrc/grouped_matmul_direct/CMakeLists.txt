include(ascend)

# Build the direct-2560 operator from source as part of the Python wheel build.
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_TOOLKIT_HOME}")
set(ASCEND_COMPUTE_UNIT ascend910_93)
set(ASCEND_PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
add_subdirectory(opp)

add_library(_grouped_matmul_direct MODULE grouped_matmul_direct.cpp)
add_dependencies(_grouped_matmul_direct cust_opapi)
find_library(TORCH_PYTHON_LIBRARY torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  REQUIRED
  NO_DEFAULT_PATH
)
set_target_properties(_grouped_matmul_direct PROPERTIES
  PREFIX ""
  INSTALL_RPATH "\$ORIGIN/grouped_matmul_direct/op_api/lib"
)

target_compile_definitions(_grouped_matmul_direct PRIVATE
  TORCH_EXTENSION_NAME=_grouped_matmul_direct
  _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}
)
target_compile_options(_grouped_matmul_direct PRIVATE -O2 -Wall -Wextra)
target_include_directories(_grouped_matmul_direct PRIVATE
  ${DLINFER_GMM_OPP_BINARY_DIR}/autogen
  ${TORCH_INCLUDE_DIRS}
  ${CANN_INCLUDE_DIRS}
  ${CANN_INCLUDE_DIRS}/aclnn
)
target_link_libraries(_grouped_matmul_direct PRIVATE
  Python::Python
  torch
  ${TORCH_PYTHON_LIBRARY}
  cust_opapi
  ${CANN_LIBRARIES}
)

install(TARGETS _grouped_matmul_direct
  LIBRARY DESTINATION dlinfer/vendor/ascend
)
install(TARGETS cust_opapi
  LIBRARY DESTINATION dlinfer/vendor/ascend/grouped_matmul_direct/op_api/lib
)
install(FILES ${DLINFER_GMM_OPP_BINARY_DIR}/autogen/aclnn_dlinfer_grouped_matmul_direct.h
  DESTINATION dlinfer/vendor/ascend/grouped_matmul_direct/op_api/include
)
