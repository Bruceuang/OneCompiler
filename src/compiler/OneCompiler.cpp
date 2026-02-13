#include "OneCompiler/OneCompiler.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace OneCompiler {

Compiler::Compiler() 
    : context(mlir::MLIRContext()),
      torchImporter(std::make_unique<Frontend::TorchImporter>(&context)) {
    registerAllDialects();
    registerAllPasses();
}

Compiler::~Compiler() = default;

mlir::LogicalResult Compiler::compile(const std::string& inputFile, 
                                        const std::string& outputFile) {
    auto endsWith = [](const std::string& str, const std::string& suffix) {
        if (suffix.size() > str.size()) return false;
        return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
    };
    
    if (endsWith(inputFile, ".pt") || endsWith(inputFile, ".pth") || 
        endsWith(inputFile, ".onnx")) {
        if (failed(importFromTorchModel(inputFile))) {
            return mlir::failure();
        }
    } else {
        llvm::errs() << "Unsupported input format: " << inputFile << "\n";
        return mlir::failure();
    }
    
    return mlir::success();
}

mlir::LogicalResult Compiler::importFromTorchModel(const std::string& modelPath) {
    return torchImporter->importFromFile(modelPath, module);
}

mlir::LogicalResult Compiler::importFromONNXModel(const std::string& onnxPath) {
    return torchImporter->importFromONNX(onnxPath, module);
}

void Compiler::registerAllDialects() {
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
}

void Compiler::registerAllPasses() {
}

} // namespace OneCompiler
