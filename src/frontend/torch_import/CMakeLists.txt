add_library(torch_import_lib
    TorchImporter.cpp
)

target_link_libraries(torch_import_lib
    PRIVATE
        MLIRIR
        MLIRParser
        MLIRSupport
        LLVMSupport
)

target_include_directories(torch_import_lib
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/llvm-project/llvm/include
        ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include
        ${CMAKE_BINARY_DIR}/third_party/llvm-project/include
        ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/include
        ${CMAKE_BINARY_DIR}/third_party/llvm-project/mlir/include
        ${CMAKE_BINARY_DIR}/third_party/llvm-project/tools/mlir/include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

separate_arguments(LLVM_DEFINITIONS_LIST UNIX_COMMAND "${LLVM_DEFINITIONS}")
foreach(def ${LLVM_DEFINITIONS_LIST})
    if(def MATCHES "^-D(.+)$")
        target_compile_definitions(torch_import_lib PUBLIC ${CMAKE_MATCH_1})
    endif()
endforeach()

target_compile_definitions(torch_import_lib PUBLIC LLVM_ENABLE_ABI_BREAKING_CHECKS=1)

if(WITH_TORCH_MLIR AND EXISTS "${TORCH_MLIR_SOURCE_DIR}")
    message(STATUS "🔍 发现Torch-MLIR: ${TORCH_MLIR_SOURCE_DIR}")
    target_compile_definitions(torch_import_lib PRIVATE WITH_TORCH_MLIR)
    target_include_directories(torch_import_lib PRIVATE
        ${TORCH_MLIR_SOURCE_DIR}
        ${TORCH_MLIR_SOURCE_DIR}/include
    )
    message(STATUS "✅ Torch-MLIR集成已启用")
else()
    message(STATUS "⚠️ Torch-MLIR不可用")
endif()

target_compile_features(torch_import_lib PRIVATE cxx_std_17)
