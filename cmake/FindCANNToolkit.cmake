include(FindPackageHandleStandardArgs)

if (DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME}
        CACHE STRING "ascend toolkit default home")
else()
    set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest"
        CACHE STRING "ascend toolkit default home")
endif()

# Include directories.
find_path(CANN_INCLUDE_DIRS
    NAMES acl/acl.h hccl/hccl.h
    PATHS ${ASCEND_TOOLKIT_HOME}/include
)

# Library dependencies.
find_library(HCCL_LIB
    NAMES hccl
    PATHS ${ASCEND_TOOLKIT_HOME}/lib64
)
if (HCCL_LIB)
    list(APPEND CANN_LIBRARY ${HCCL_LIB})
else()
    message(FATAL_ERROR "libhccl.so not found")
endif()

find_library(OPAPI_LIB
    NAMES opapi
    PATHS ${ASCEND_TOOLKIT_HOME}/lib64
)
if (OPAPI_LIB)
    list(APPEND CANN_LIBRARY ${OPAPI_LIB})
else()
    message(FATAL_ERROR "libopapi.so not found")
endif()

set(CANN_LIBRARIES ${CANN_LIBRARY})

#TODO (chenchiyu): construct modern cmake target for CANNToolkit
message(STATUS "Found CANN Toolkit: CANN_LIBRARIES: ${CANN_LIBRARIES}, CANN_INCLUDE_DIRS: ${CANN_INCLUDE_DIRS}")
find_package_handle_standard_args(CANNToolkit DEFAULT_MSG CANN_LIBRARIES CANN_INCLUDE_DIRS)
