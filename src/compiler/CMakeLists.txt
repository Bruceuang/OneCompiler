# # OneCompiler核心实现
# target_sources(OneCompiler
#     PRIVATE
#     OneCompiler.cpp
# )

# # 设置包含目录
# target_include_directories(OneCompilerImpl
#     PUBLIC
#     ${CMAKE_SOURCE_DIR}/include
#     ${LLVM_INCLUDE_DIRS}
#     ${MLIR_INCLUDE_DIRS}
# )

# # 链接必要的LLVM/MLIR库
# target_link_libraries(OneCompilerImpl
#     PUBLIC
#     MLIRIR
#     MLIRSupport
#     MLIRDialect
# )

# 添加compiler子目录 
# add_subdirectory(compiler)
# 其他目录暂时注释掉，等它们有内容时再启用 
# add_subdirectory(dialects) 
# add_subdirectory(transforms) 
# add_subdirectory(tools)

# compiler子目录的CMakeLists.txt
# OneCompiler.cpp的源文件已经在父目录中添加
# 这里可以添加compiler特定的配置