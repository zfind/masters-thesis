cmake_minimum_required(VERSION 3.9)
project(gp)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61)

cuda_add_executable(gp main.cpp)