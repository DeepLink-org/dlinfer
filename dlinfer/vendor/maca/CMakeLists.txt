cmake_minimum_required(VERSION 3.21)

find_package(PythonInterp REQUIRED)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print('{}{}'.format(sys.version_info.major, sys.version_info.minor))"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND python -c "from torch.utils import cmake_prefix_path; \
    print(cmake_prefix_path + '/Torch', end='')"
    OUTPUT_VARIABLE Torch_DIR
)

include(ExternalProject)

# environment variable
set(ENV{MACA_PATH} "/opt/maca")
set(ENV{CUDA_PATH} "/usr/local/cuda")
set(ENV{CUCC_PATH} "$ENV{MACA_PATH}/tools/cu-bridge")
set(ENV{PATH} "$ENV{CUDA_PATH}/bin:$ENV{MACA_PATH}/mxgpu_llvm/bin:$ENV{MACA_PATH}/bin:$ENV{CUCC_PATH}/tools:$ENV{CUCC_PATH}/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "$ENV{MACA_PATH}/lib:$ENV{MACA_PATH}/mxgpu_llvm/lib:$ENV{LD_LIBRARY_PATH}")

set(MACA_SUB_MODULE maca_extension)
set(MACA_SUB_MODULE_BINRAY_DIR "${PROJECT_BINARY_DIR}/${MACA_SUB_MODULE}")
set(MACA_SUB_MODULE_BYPRODUCTS "${MACA_SUB_MODULE_BINRAY_DIR}/${MACA_SUB_MODULE}.cpython-${PYTHON_VERSION}-x86_64-linux-gnu.so")

ExternalProject_Add(${MACA_SUB_MODULE}
                    SOURCE_DIR "${PROJECT_SOURCE_DIR}/dlinfer/vendor/maca/csrc"
                    BINARY_DIR "${MACA_SUB_MODULE_BINRAY_DIR}"
                    CMAKE_COMMAND "cmake_maca"
                    BUILD_BYPRODUCTS "${MACA_SUB_MODULE_BYPRODUCTS}"
                    BUILD_ALWAYS ON
                    USES_TERMINAL_BUILD ON
                    USES_TERMINAL_INSTALL ON
                    CMAKE_ARGS
                      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                      "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
                      "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
                      "-DCMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS}"
                      "-DTorch_DIR=${Torch_DIR}"
                      "-DUSE_MACA=ON")

install(
  FILES ${MACA_SUB_MODULE_BYPRODUCTS}
  DESTINATION ./dlinfer/vendor/maca
)
