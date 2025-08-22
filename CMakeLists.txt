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

list(APPEND SUPPORTED_DEVICE "ascend" "maca" "camb" "ppu")

if(NOT DEVICE)
  message(FATAL_ERROR "Please specify variable DEVICE of dlinfer!")
elseif(NOT DEVICE IN_LIST SUPPORTED_DEVICE)
  message(FATAL_ERROR "Device ${DEVICE} is not supported! Supported devices: ${SUPPORTED_DEVICE}")
endif()

add_subdirectory(dlinfer/vendor/${DEVICE})
add_subdirectory(dlinfer/graph/dicp/vendor)
