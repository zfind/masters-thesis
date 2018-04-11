cmake_minimum_required(VERSION 3.8)
project(GPSymbReg)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

find_package(CUDA REQUIRED)

SET(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,/usr/local/lib")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61 --std=c++11)


set(SOURCE_FILES
        main.cpp
        SymbRegEvalOp.cpp
        SymbRegEvalOp.h
        CudaEvaluator.h)

cuda_add_library(CudaEvaluator_o CudaEvaluator.cu)

cuda_add_executable(GPSymbReg ${SOURCE_FILES})

target_link_libraries(GPSymbReg ${CUDA_LIBRARIES})

target_link_libraries(GPSymbReg CudaEvaluator_o)

target_link_libraries(GPSymbReg ecf)
