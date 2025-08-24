include_guard()

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
                COMMAND ${CMAKE_COMMAND} --build ${LLVM_BUILD_DIR} --target all --parallel
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