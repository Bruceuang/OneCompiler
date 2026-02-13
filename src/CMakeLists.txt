find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

llvm_map_components_to_libnames(llvm_libs Support Core)

add_library(one_compiler_lib
    compiler/OneCompiler.cpp
)

target_include_directories(one_compiler_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/llvm-project/llvm/include
        ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include
        ${CMAKE_BINARY_DIR}/third_party/llvm-project/include
        ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/include
        ${CMAKE_BINARY_DIR}/third_party/llvm-project/mlir/include
        ${CMAKE_BINARY_DIR}/third_party/llvm-project/tools/mlir/include
)

target_link_libraries(one_compiler_lib
    PUBLIC
        MLIRIR
        MLIRSupport
        MLIRDialect
        MLIRFuncDialect
        MLIRArithDialect
        MLIRParser
        LLVMCore
        LLVMSupport
)

separate_arguments(LLVM_DEFINITIONS_LIST UNIX_COMMAND "${LLVM_DEFINITIONS}")
foreach(def ${LLVM_DEFINITIONS_LIST})
    if(def MATCHES "^-D(.+)$")
        target_compile_definitions(one_compiler_lib PUBLIC ${CMAKE_MATCH_1})
    endif()
endforeach()

target_compile_definitions(one_compiler_lib PUBLIC LLVM_ENABLE_ABI_BREAKING_CHECKS=1)

add_subdirectory(frontend)
add_subdirectory(tools)
