cmake_minimum_required(VERSION 2.8)
project(GPSymbReg)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

SET(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,/usr/local/lib")

set(SOURCE_FILES
    main.cpp
    SymbRegEvalOp.cpp
    SymbRegEvalOp.h)

#include_directories(/usr/local/include)
add_executable(GPSymbReg ${SOURCE_FILES})

target_link_libraries(GPSymbReg ecf)


#
#if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#    target_link_libraries(GPSymbReg ${CMAKE_CURRENT_SOURCE_DIR}/../../debug/libECF.a)
#endif()
#if(CMAKE_BUILD_TYPE STREQUAL "Release")
#    target_link_libraries(GPSymbReg ${CMAKE_CURRENT_SOURCE_DIR}/../../release/libECF.a)
#endif()