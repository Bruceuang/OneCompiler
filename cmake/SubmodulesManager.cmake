include_guard()

# 检查Torch-MLIR子模块状态
function(check_torch_mlir_submodule RESULT_VAR)
    set(TORCH_MLIR_FOUND FALSE)
    
    # 检查指定的Torch-MLIR路径是否存在
    if(EXISTS "${TORCH_MLIR_SOURCE_DIR}")
        set(TORCH_MLIR_FOUND TRUE)
    else()
        message(STATUS "⚠️  Torch-MLIR未找到: ${TORCH_MLIR_SOURCE_DIR}")
        message(STATUS "💡 建议设置正确的路径: -DTORCH_MLIR_PATH=path/to/torch-mlir")
    endif()
    
    set(${RESULT_VAR} ${TORCH_MLIR_FOUND} PARENT_SCOPE)
endfunction()

# 构建Torch-MLIR (如果需要)
function(build_torch_mlir_submodule)
    # 检查Torch-MLIR状态
    check_torch_mlir_submodule(TORCH_MLIR_OK)
    if(NOT TORCH_MLIR_OK)
        message(WARNING "⚠️ Torch-MLIR不可用")
        return()
    endif()
    
    # 检查是否需要构建 (简化版，假设用户已安装torch-mlir)
    message(STATUS "✅ Torch-MLIR路径已配置: ${TORCH_MLIR_SOURCE_DIR}")
endfunction()

# 检查LLVM子模块状态
function(check_llvm_submodule RESULT_VAR)
    set(LLVM_SUBMODULE_PATH "${ONECOMPILER_SUBMODULE_ROOT}/llvm-project")
    
    # 检查子模块目录是否存在
    if(NOT EXISTS "${LLVM_SUBMODULE_PATH}")
        message(STATUS "⚠️  LLVM子模块未找到")
        message(STATUS "💡 建议运行: git submodule update --init --recursive")
        message(STATUS "💡 或使用: ./scripts/init_submodules.sh")
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    
    # 检查是否为git子模块
    if(NOT EXISTS "${LLVM_SUBMODULE_PATH}/.git")
        message(STATUS "⚠️  LLVM子模块未正确初始化")
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
    endif()
    
    set(${RESULT_VAR} TRUE PARENT_SCOPE)
endfunction()

# 构建LLVM子模块
function(build_llvm_submodule)
    # 检查LLVM子模块状态
    check_llvm_submodule(LLVM_OK)
    if(NOT LLVM_OK)
        message(FATAL_ERROR "❌ LLVM子模块未初始化")
    endif()
    
    # 检查是否需要构建
    if(NOT EXISTS "${LLVM_BUILD_DIR}/CMakeCache.txt")
        message(STATUS "🔧 构建LLVM/MLIR...")
        message(STATUS "💡 使用内存优化配置...")
        
        # 获取CPU核心数，但限制最大并行度
        include(ProcessorCount)
        ProcessorCount(N)
        if(NOT N EQUAL 0)
            # 限制并行进程数以减少内存使用
            math(EXPR LLVM_PARALLEL_JOBS "${N}/2")
            if(LLVM_PARALLEL_JOBS LESS 1)
                set(LLVM_PARALLEL_JOBS 1)
            endif()
            if(LLVM_PARALLEL_JOBS GREATER 4)
                set(LLVM_PARALLEL_JOBS 4)  # 最大限制为4
            endif()
        else()
            set(LLVM_PARALLEL_JOBS 2)  # 默认2个进程
        endif()
        
        message(STATUS "⚙️  并行构建进程数: ${LLVM_PARALLEL_JOBS}")
        
        # 执行构建
        execute_process(
            COMMAND ${CMAKE_COMMAND}
                -G ${CMAKE_GENERATOR}
                ${LLVM_CMAKE_ARGS}
                -S ${LLVM_SOURCE_DIR}
                -B ${LLVM_BUILD_DIR}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            RESULT_VARIABLE LLVM_BUILD_RESULT
        )
        
        if(LLVM_BUILD_RESULT EQUAL 0)
            execute_process(
                COMMAND ${CMAKE_COMMAND} --build ${LLVM_BUILD_DIR} 
                    --target all 
                    --parallel ${LLVM_PARALLEL_JOBS}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                RESULT_VARIABLE LLVM_BUILD_RESULT
            )
        endif()
        
        if(NOT LLVM_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "❌ LLVM构建失败")
        else()
            message(STATUS "✅ LLVM构建完成")
        endif()
    else()
        message(STATUS "✅ LLVM已构建")
    endif()
endfunction()

# 显示帮助信息
function(show_llvm_help)
    message(STATUS "📋 LLVM子模块管理:")
    message(STATUS "  初始化: git submodule update --init --recursive")
    message(STATUS "  强制更新: git submodule update --init --recursive --force")
endfunction()