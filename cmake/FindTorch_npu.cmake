include(FindPackageHandleStandardArgs)

# Include directories.
find_path(TORCH_NPU_INCLUDE_DIRS NAMES torch_npu/csrc/include/ops.h)

# Library dependencies.
find_library(TORCH_NPU_LIBRARY NAMES torch_npu npu_profiler)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'")
endif()
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_LIBRARY})

# torch/csrc/python_headers depends Python.h
find_package(Python COMPONENTS Interpreter Development)

#TODO (chenchiyu): construct modern cmake target for Torch_npu
message(STATUS "Found Torch_npu: TORCH_NPU_LIBRARY: ${TORCH_NPU_LIBRARY}, TORCH_NPU_INCLUDE_DIRS: ${TORCH_NPU_INCLUDE_DIRS}")
find_package_handle_standard_args(Torch_npu DEFAULT_MSG TORCH_NPU_LIBRARY TORCH_NPU_INCLUDE_DIRS)
