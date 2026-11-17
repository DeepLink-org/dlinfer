include(ascend)

set(GMM_DIRECT_VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
set(GMM_DIRECT_OPAPI ${GMM_DIRECT_VENDOR_DIR}/op_api/lib/libcust_opapi.so)

if(NOT EXISTS ${GMM_DIRECT_OPAPI})
  message(FATAL_ERROR "Missing bundled direct2560 op-api: ${GMM_DIRECT_OPAPI}")
endif()

add_library(_grouped_matmul_direct MODULE grouped_matmul_direct.cpp)
find_library(TORCH_PYTHON_LIBRARY torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  REQUIRED
  NO_DEFAULT_PATH
)
set_target_properties(_grouped_matmul_direct PROPERTIES
  PREFIX ""
  BUILD_RPATH "${GMM_DIRECT_VENDOR_DIR}/lib"
  INSTALL_RPATH "\$ORIGIN/grouped_matmul_direct/op_api/lib"
)

target_compile_definitions(_grouped_matmul_direct PRIVATE
  TORCH_EXTENSION_NAME=_grouped_matmul_direct
  _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}
)
target_compile_options(_grouped_matmul_direct PRIVATE -O2 -Wall -Wextra)
target_include_directories(_grouped_matmul_direct PRIVATE
  ${GMM_DIRECT_VENDOR_DIR}/op_api/include
  ${TORCH_INCLUDE_DIRS}
  ${CANN_INCLUDE_DIRS}
  ${CANN_INCLUDE_DIRS}/aclnn
)
target_link_libraries(_grouped_matmul_direct PRIVATE
  Python::Python
  torch
  ${TORCH_PYTHON_LIBRARY}
  ${GMM_DIRECT_OPAPI}
  ${CANN_LIBRARIES}
)

install(TARGETS _grouped_matmul_direct
  LIBRARY DESTINATION dlinfer/vendor/ascend
)
install(DIRECTORY ${GMM_DIRECT_VENDOR_DIR}/
  DESTINATION dlinfer/vendor/ascend/grouped_matmul_direct
)
