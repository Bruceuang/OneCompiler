# 查找LLVM和MLIR包
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# 包含LLVM和MLIR的包含目录
llvm_map_components_to_libnames(llvm_libs Support Core)
separate_arguments(LLVM_DEFINITIONS_LIST UNIX_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFINITIONS_LIST})

# OneCompiler主库
add_library(OneCompiler
    compiler/OneCompiler.cpp
)

# 设置包含目录 - 使用绝对路径
target_include_directories(OneCompiler
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/llvm-project/llvm/include
    ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include
    ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/include
    ${CMAKE_BINARY_DIR}/third_party/llvm-project/mlir/include
)

# 添加依赖确保TableGen文件先生成
add_dependencies(OneCompiler
    MLIRIR
    MLIRBuiltinOpsIncGen
    MLIRBuiltinLocationAttributesIncGen
)

# 链接必要的LLVM/MLIR库
target_link_libraries(OneCompiler
    PUBLIC
    MLIRIR
    MLIRSupport
    MLIRDialect
    MLIRFuncDialect
    MLIRArithDialect
    LLVMCore
    LLVMSupport
)

# 添加compiler子目录
add_subdirectory(compiler)

# 其他目录暂时注释掉，等它们有内容时再启用
# add_subdirectory(dialects)
# add_subdirectory(transforms)
# add_subdirectory(tools)