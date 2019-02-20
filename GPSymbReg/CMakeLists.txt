cmake_minimum_required(VERSION 3.8)
project(GPSymbReg)

set(CMAKE_CXX_STANDARD 14)

find_package(Boost)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -gencode arch=compute_61,code=sm_61
)

set(SOURCE_FILES
        src/main.cpp
        src/Constants.h
        src/Dataset.h
        src/Dataset.cpp
        src/SimpleEvaluator.h
        src/SimpleEvaluator.cpp
        src/PostfixEvaluator.h
        src/PostfixEvaluator.cpp
        src/CUPostfixEvalOp.h
        src/CUPostfixEvalOp.cu
        src/BenchmarkOp.h
        src/BenchmarkOp.cpp
)


add_subdirectory(lib/ECF)

#cuda_add_library(CudaEvaluator_o src/CudaEvaluator.cu)

cuda_add_executable(GPSymbReg ${SOURCE_FILES})

target_include_directories(GPSymbReg PRIVATE lib/ECF)

target_link_libraries(GPSymbReg ECF)

target_link_libraries(GPSymbReg ${CUDA_LIBRARIES})

#target_link_libraries(GPSymbReg CudaEvaluator_o)
