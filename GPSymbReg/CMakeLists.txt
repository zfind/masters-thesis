cmake_minimum_required(VERSION 3.8)
project(GPSymbReg)

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
    src/Constants.h
    src/CudaEvaluator.h
)

add_subdirectory(lib/ECF)

cuda_add_library(CudaEvaluator_o src/CudaEvaluator.cu)

cuda_add_executable(GPSymbReg ${SOURCE_FILES})

target_include_directories(GPSymbReg PRIVATE lib/ECF)

target_link_libraries(GPSymbReg ECF)

target_link_libraries(GPSymbReg ${CUDA_LIBRARIES})

target_link_libraries(GPSymbReg CudaEvaluator_o)

