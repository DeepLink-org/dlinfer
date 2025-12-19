cmake_minimum_required(VERSION 3.18)
project(dlinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# the default CMAKE_BUILD_TYPE is Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(DEVICE "" CACHE STRING "device string, default empty string")
string(TOLOWER "${DEVICE}" DEVICE)

list(APPEND SUPPORTED_DEVICE "ascend" "maca" "camb")

if(NOT DEVICE)
  message(FATAL_ERROR "Please specify variable DEVICE of dlinfer!")
elseif(NOT DEVICE IN_LIST SUPPORTED_DEVICE)
  message(FATAL_ERROR "Device ${DEVICE} is not supported! Supported devices: ${SUPPORTED_DEVICE}")
endif()

#
# Optional: build DICP vendor (e.g. AtbGraph runtime for ascend).
# Default OFF. Turn on via env: DLINFER_BUILD_DICP=ON/1/TRUE/YES
# or via cmake: -DDLINFER_BUILD_DICP=ON
#
option(DLINFER_BUILD_DICP "Build DICP vendor components" OFF)

if(DEFINED ENV{DLINFER_BUILD_DICP} AND NOT "$ENV{DLINFER_BUILD_DICP}" STREQUAL "")
  string(TOUPPER "$ENV{DLINFER_BUILD_DICP}" _DLINFER_BUILD_DICP_ENV)
  if(_DLINFER_BUILD_DICP_ENV STREQUAL "1"
     OR _DLINFER_BUILD_DICP_ENV STREQUAL "ON"
     OR _DLINFER_BUILD_DICP_ENV STREQUAL "TRUE"
     OR _DLINFER_BUILD_DICP_ENV STREQUAL "YES")
    set(DLINFER_BUILD_DICP ON CACHE BOOL "Build DICP vendor components" FORCE)
  endif()
endif()

if(DLINFER_BUILD_DICP)
  message(STATUS "DLINFER_BUILD_DICP=ON: enabling DICP vendor build")
  add_subdirectory(dlinfer/graph/dicp/vendor)
else()
  message(STATUS "DLINFER_BUILD_DICP=OFF: skipping DICP vendor build (set env DLINFER_BUILD_DICP=ON to enable)")
endif()

install(CODE "message(STATUS \"Install completed for device: ${DEVICE}\")")
