# MLIR测试
add_lit_testsuite(check-one-mlir "Running OneCompiler MLIR tests"
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
    OneCompiler
    FileCheck count not
)