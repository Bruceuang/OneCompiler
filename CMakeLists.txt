cmake_minimum_required(VERSION 3.20.0)
project(OneCompiler VERSION 0.0.1 LANGUAGES CXX C)

# 基本设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 包含子模块配置
include(cmake/SubmodulesConfig.cmake)
include(cmake/SubmodulesManager.cmake)

# 检查子模块状态
message(STATUS "🔍 检查LLVM子模块状态...") 
check_llvm_submodule(LLVM_STATUS) 
if(NOT LLVM_STATUS) 
    message(STATUS "❌ LLVM子模块未初始化") 
    show_llvm_help() 
    message(FATAL_ERROR "❌ LLVM子模块未初始化，请先初始化LLVM子模块") 
endif()

# 构建LLVM子模块
message(STATUS "🔧 构建LLVM/MLIR...") 
build_llvm_submodule(LLVM_OK)
if(NOT LLVM_OK)
    message(FATAL_ERROR "❌ LLVM子模块构建失败")
endif()

# 设置LLVM和MLIR路径
set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm") 
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir") 
message(STATUS "✅ LLVM路径: ${LLVM_DIR}") 
message(STATUS "✅ MLIR路径: ${MLIR_DIR}")

# 将路径添加到CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${LLVM_DIR}") 
list(APPEND CMAKE_PREFIX_PATH "${MLIR_DIR}")

# 添加子目录
add_subdirectory(src)

# 添加测试和示例
if(EXISTS ${CMAKE_SOURCE_DIR}/test/CMakeLists.txt)
    add_subdirectory(test)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/examples/CMakeLists.txt)
    add_subdirectory(examples)
endif()