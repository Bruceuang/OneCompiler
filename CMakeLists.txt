cmake_minimum_required(VERSION 3.20.0)
project(OneCompiler VERSION 0.0.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(ONECOMPILER_BUILD_EXAMPLES "Build example programs" ON)
option(ONECOMPILER_BUILD_TESTS "Build tests" OFF)
option(WITH_TORCH_MLIR "Enable torch-mlir integration" OFF)

include(cmake/SubmodulesConfig.cmake)
include(cmake/SubmodulesManager.cmake)

message(STATUS "🔍 检查LLVM子模块状态...") 
check_llvm_submodule(LLVM_STATUS) 
if(NOT LLVM_STATUS) 
    message(STATUS "❌ LLVM子模块未初始化") 
    show_llvm_help() 
    message(FATAL_ERROR "❌ LLVM子模块未初始化，请先初始化LLVM子模块") 
endif()

build_torch_mlir_submodule()

message(STATUS "🔧 构建LLVM/MLIR...") 
build_llvm_submodule()

set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm") 
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir") 
message(STATUS "✅ LLVM路径: ${LLVM_DIR}") 
message(STATUS "✅ MLIR路径: ${MLIR_DIR}")

list(APPEND CMAKE_PREFIX_PATH "${LLVM_DIR}") 
list(APPEND CMAKE_PREFIX_PATH "${MLIR_DIR}")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

set(LLVM_INCLUDE_DIRS "${LLVM_BUILD_DIR}/include")
set(MLIR_INCLUDE_DIRS "${LLVM_BUILD_DIR}/include")

add_subdirectory(src)

if(ONECOMPILER_BUILD_EXAMPLES AND EXISTS ${CMAKE_SOURCE_DIR}/examples/CMakeLists.txt)
    add_subdirectory(examples)
endif()

if(ONECOMPILER_BUILD_TESTS AND EXISTS ${CMAKE_SOURCE_DIR}/test/CMakeLists.txt)
    add_subdirectory(test)
endif()