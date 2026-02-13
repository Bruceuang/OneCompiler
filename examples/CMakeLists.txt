add_executable(torch_import_example
    torch_import_example.cpp
)

target_include_directories(torch_import_example PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/llvm-project/llvm/include
    ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include
    ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/include
    ${CMAKE_BINARY_DIR}/third_party/llvm-project/mlir/include
    ${CMAKE_BINARY_DIR}/third_party/llvm-project/tools/mlir/include
)

target_link_libraries(torch_import_example PRIVATE
    one_compiler_lib
    torch_import_lib
    MLIRIR
    MLIRParser
    MLIRSupport
    LLVMSupport
)

target_compile_features(torch_import_example PRIVATE cxx_std_17)
