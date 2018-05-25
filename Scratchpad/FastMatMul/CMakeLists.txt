cmake_minimum_required(VERSION 3.8)

project(CudaMatMul)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61 --std=c++14)

set(SOURCE_FILES
        main.cpp
        Matrix.h
        Matrix.cpp
        Matrix.cu)


cuda_add_library(CudaMatMul_o Matrix.cu)

cuda_add_executable(CudaMatMul main.cpp ${SOURCE_FILES})

target_link_libraries(CudaMatMul ${CUDA_LIBRARIES})

target_link_libraries(CudaMatMul CudaMatMul_o)