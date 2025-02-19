cmake_minimum_required(VERSION 3.21)

set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

project(maca_extension LANGUAGES CXX)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
option(USE_MACA "enbale MACA" ON)

find_package(Torch REQUIRED)
add_compile_definitions(USE_MACA)
find_package(Python COMPONENTS Interpreter Development.Module)

find_library(torch_python_LIBRARY torch_python PATHS
  "${TORCH_INSTALL_PREFIX}/lib")

set(DLINFER_VLLM_SRC
  "pybind.cpp"
  "cache_kernels.cu"
  "attention/attention_kernels.cu"
  "pos_encoding_kernels.cu"
  "moe_align_block_size_kernels.cu"
  "moe/topk_softmax_kernels.cu"
)

Python_add_library(
  ${PROJECT_NAME} MODULE WITH_SOABI 
  ${DLINFER_VLLM_SRC}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  "-DTORCH_EXTENSION_NAME=${PROJECT_NAME}")

target_link_libraries(${PROJECT_NAME} PRIVATE 
  torch ${torch_python_LIBRARY}
  mcblas
)

# get conflict options in CMAKE_CUDA_FLAGS
string(REGEX MATCH "-mllvm -metaxgpu-disable-bsm-offset=[0-9]+" TORCH_METAXGPU_DISABLE_BSM_OFFSET_FLAGS ${CMAKE_CUDA_FLAGS})
string(REGEX MATCH "-mllvm -metaxgpu-inlinescope=[0-9]+" TORCH_METAXGPU_INLINESCOPE_FLAGS ${CMAKE_CUDA_FLAGS})
string(REPLACE " " ";" TORCH_METAXGPU_DISABLE_BSM_OFFSET_FLAGS ${TORCH_METAXGPU_DISABLE_BSM_OFFSET_FLAGS})
string(REPLACE " " ";" TORCH_METAXGPU_INLINESCOPE_FLAGS ${TORCH_METAXGPU_INLINESCOPE_FLAGS})
# remove conflict options in CMAKE_CUDA_FLAGS
string(REGEX REPLACE "-mllvm -metaxgpu-disable-bsm-offset=[0-9]+" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
string(REGEX REPLACE "-mllvm -metaxgpu-inlinescope=[0-9]+" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})

# set vllm files properties
set(DLINFER_VLLM_COMPILE_OPTIONS
  "-D__CUDA_NO_HALF_OPERATORS__"
  "-D__MACA_NO_HALF_OPERATORS__"
  "-D__CUDA_NO_HALF_CONVERSIONS__"
  "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
  "-D__CUDA_NO_HALF2_OPERATORS__"
  ${TORCH_METAXGPU_DISABLE_BSM_OFFSET_FLAGS}
  ${TORCH_METAXGPU_INLINESCOPE_FLAGS}
)

set_source_files_properties(${DLINFER_VLLM_SRC} PROPERTIES
  COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${DLINFER_VLLM_COMPILE_OPTIONS}>"
  INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}"
)
