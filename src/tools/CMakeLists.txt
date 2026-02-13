add_executable(onecompiler
    onecompiler.cpp
)

target_include_directories(onecompiler PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/llvm-project/llvm/include
    ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include
    ${CMAKE_BINARY_DIR}/third_party/llvm-project/include
    ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/include
    ${CMAKE_BINARY_DIR}/third_party/llvm-project/mlir/include
    ${CMAKE_BINARY_DIR}/third_party/llvm-project/tools/mlir/include
)

target_link_libraries(onecompiler PRIVATE
    one_compiler_lib
    torch_import_lib
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRPass
    LLVMSupport
)

separate_arguments(LLVM_DEFINITIONS_LIST UNIX_COMMAND "${LLVM_DEFINITIONS}")
foreach(def ${LLVM_DEFINITIONS_LIST})
    if(def MATCHES "^-D(.+)$")
        target_compile_definitions(onecompiler PUBLIC ${CMAKE_MATCH_1})
    endif()
endforeach()

target_compile_definitions(onecompiler PUBLIC LLVM_ENABLE_ABI_BREAKING_CHECKS=1)

target_compile_features(onecompiler PRIVATE cxx_std_17)
