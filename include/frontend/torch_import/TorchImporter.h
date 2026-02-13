#ifndef ONECOMPILER_FRONTEND_TORCHIMPORTER_H
#define ONECOMPILER_FRONTEND_TORCHIMPORTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include <string>
#include <memory>

namespace OneCompiler {
namespace Frontend {

class TorchImporter {
public:
    TorchImporter(mlir::MLIRContext *context);
    ~TorchImporter();

    mlir::LogicalResult importFromFile(const std::string &modelPath, 
                                      mlir::OwningOpRef<mlir::ModuleOp> &module);

    mlir::LogicalResult importFromONNX(const std::string &onnxPath, 
                                      mlir::OwningOpRef<mlir::ModuleOp> &module);

    void setVerbose(bool verbose);
    void setEnableOptimizations(bool enable);
    void setTargetDialect(const std::string &dialectName);

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Frontend
} // namespace OneCompiler

#endif // ONECOMPILER_FRONTEND_TORCHIMPORTER_H
