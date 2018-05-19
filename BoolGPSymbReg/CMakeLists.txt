cmake_minimum_required(VERSION 3.9)
project(BoolGPSymbReg)

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
        Constants.h
        Primitives.cpp
        SymbRegEvalOp.cpp
        SymbRegEvalOp.h
        CudaEvaluator.h
        )

cuda_add_library(CudaEvaluator_o CudaEvaluator.cu)

cuda_add_executable(BoolGPSymbReg ${SOURCE_FILES})

target_link_libraries(BoolGPSymbReg ${CUDA_LIBRARIES})

target_link_libraries(BoolGPSymbReg CudaEvaluator_o)

target_link_libraries(BoolGPSymbReg ecf)
