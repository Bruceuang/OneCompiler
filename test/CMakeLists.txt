# OneCompiler测试套件

# 查找LLVM的lit测试工具
find_program(LLVM_LIT_EXECUTABLE
    NAMES lit lit.py
    PATHS ${LLVM_SOURCE_DIR}/utils/lit
    REQUIRED
)

# 添加MLIR测试
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mlir)
    add_subdirectory(mlir)
endif()

# 添加Python测试
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/python)
    add_subdirectory(python)
endif()

# 配置lit测试套件
if(LLVM_LIT_EXECUTABLE)
    configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    )
endif()