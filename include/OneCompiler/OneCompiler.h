#ifndef ONECOMPILER_ONECOMPILER_H
#define ONECOMPILER_ONECOMPILER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "frontend/torch_import/TorchImporter.h"
#include <memory>

namespace OneCompiler {

class Compiler {
public:
    Compiler();
    ~Compiler();

    mlir::LogicalResult compile(const std::string& inputFile, 
                               const std::string& outputFile);

    mlir::LogicalResult importFromTorchModel(const std::string& modelPath);
    mlir::LogicalResult importFromONNXModel(const std::string& onnxPath);

    void registerAllDialects();
    void registerAllPasses();

private:
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::unique_ptr<Frontend::TorchImporter> torchImporter;
};

} // namespace OneCompiler

#endif // ONECOMPILER_ONECOMPILER_H
