include(FindPackageHandleStandardArgs)

if (DEFINED ENV{ATB_HOME_PATH})
    set(ATB_HOME_PATH $ENV{ATB_HOME_PATH}
        CACHE STRING "atb default home")
else()
    set(ATB_HOME_PATH "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0"
        CACHE STRING "atb toolkit default home")
endif()

# Extract cxx_abi version from ATB_HOME_PATH (last path component)
get_filename_component(ATB_CXX_ABI_VERSION ${ATB_HOME_PATH} NAME)
message(STATUS "ATB_CXX_ABI_VERSION: ${ATB_CXX_ABI_VERSION}")

# Include directories.
find_path(ATB_INCLUDE_DIRS
    NAMES atb/atb_infer.h
    PATHS ${ATB_HOME_PATH}/include
)

# Library dependencies.
find_library(ATB_LIBRARY
    NAMES atb
    PATHS ${ATB_HOME_PATH}/lib
)
set(ATB_LIBRARIES ${ATB_LIBRARY})

#TODO (chenchiyu): construct modern cmake target for ATB
message(STATUS "Found ATB: ATB_LIBRARIES: ${ATB_LIBRARIES}, ATB_INCLUDE_DIRS: ${ATB_INCLUDE_DIRS}")
find_package_handle_standard_args(ATB DEFAULT_MSG ATB_LIBRARIES ATB_INCLUDE_DIRS)
