# modified from torch_npu cmake.in
include(FindPackageHandleStandardArgs)

set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ascend toolkit default home")

# Include directories.
find_path(HCCL_INCLUDE_DIRS NAMES hccl/hccl.h PATHS ${ASCEND_TOOLKIT_HOME}/include)

# Library dependencies.
find_library(HCCL_LIBRARY NAMES hccl PATHS ${ASCEND_TOOLKIT_HOME}/lib64)
set(HCCL_LIBRARIES ${HCCL_LIBRARY})

#TODO (chenchiyu): construct modern cmake target for Hccl
message(STATUS "Found Hccl: HCCL_LIBRARIES: ${HCCL_LIBRARIES}, HCCL_INCLUDE_DIRS: ${HCCL_INCLUDE_DIRS}")
find_package_handle_standard_args(Hccl DEFAULT_MSG HCCL_LIBRARIES HCCL_INCLUDE_DIRS)
