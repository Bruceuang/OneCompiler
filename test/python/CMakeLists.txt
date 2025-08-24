# Python集成测试
add_lit_testsuite(check-one-python "Running OneCompiler Python tests"
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
    OneCompiler
)