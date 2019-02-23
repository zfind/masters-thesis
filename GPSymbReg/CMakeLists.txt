cmake_minimum_required(VERSION 3.8)
project(GPSymbReg)

set(CMAKE_CXX_STANDARD 14)

find_package(Boost)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

find_package(CUDA 9.1 REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -gencode arch=compute_61,code=sm_61
        )

set(SOURCE_FILES
        src/main.cpp
        src/PostfixEvalOpUtils.h
        src/PostfixEvalOpUtils.cpp
        src/Dataset.h
        src/Dataset.cpp
        src/SymbRegEvalOp.h
        src/SymbRegEvalOp.cpp
        src/CpuPostfixEvalOp.h
        src/CpuPostfixEvalOp.cpp
        src/CudaPostfixEvalOp.h
        src/CudaPostfixEvalOp.cpp
        src/CudaPostfixEvalOp.cu
        src/BenchmarkEvalOp.h
        src/BenchmarkEvalOp.cpp
        src/Timer.h
        )


add_subdirectory(lib/ECF)

cuda_add_executable(GPSymbReg ${SOURCE_FILES})

target_include_directories(GPSymbReg PRIVATE lib/ECF)

target_link_libraries(GPSymbReg ECF)

target_link_libraries(GPSymbReg ${CUDA_LIBRARIES})