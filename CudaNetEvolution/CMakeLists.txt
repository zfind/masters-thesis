cmake_minimum_required(VERSION 3.8)

project(CudaNetEvolution)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

cuda_add_executable(CudaNetEvolution ${cpu_source_files} ${gpu_source_files})
