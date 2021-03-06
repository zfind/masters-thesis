cmake_minimum_required(VERSION 3.8)

project(CudaNetEvolution)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61 --std=c++14)

set(SOURCE_FILES
        main.cpp
        Common.h
        ClonAlg.h
        ClonAlg.cpp
        Dataset.h
        Dataset.cpp
        Matrix.h
        Matrix.cpp
        Net.h
        CudaEvaluator.h
        CudaEvaluator.cu)


cuda_add_library(CudaEvaluator_o Net.cu CudaEvaluator.cu)

cuda_add_executable(CudaNetEvolution main.cpp ${SOURCE_FILES})

target_link_libraries(CudaNetEvolution ${CUDA_LIBRARIES})

target_link_libraries(CudaNetEvolution CudaEvaluator_o)
