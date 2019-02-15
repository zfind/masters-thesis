cmake_minimum_required(VERSION 3.9)
project(BoolGPSymbReg)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -gencode arch=compute_61,code=sm_61 
    --std=c++14
)

set(SOURCE_FILES
    src/main.cpp
    src/SymbRegEvalOp.h
    src/SymbRegEvalOp.cpp
    src/Primitives.cpp
    src/Constants.h
    src/CudaEvaluator.h
)

add_subdirectory(lib/ECF)

cuda_add_library(CudaEvaluator_o src/CudaEvaluator.cu)

cuda_add_executable(BoolGPSymbReg ${SOURCE_FILES})

target_include_directories(BoolGPSymbReg PRIVATE lib/ECF)

target_link_libraries(BoolGPSymbReg ECF)

target_link_libraries(BoolGPSymbReg ${CUDA_LIBRARIES})

target_link_libraries(BoolGPSymbReg CudaEvaluator_o)

