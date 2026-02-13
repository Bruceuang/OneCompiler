#include "frontend/torch_import/TorchImporter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <algorithm>

namespace OneCompiler {
namespace Frontend {

class TorchImporter::Impl {
public:
    Impl(mlir::MLIRContext *context)
        : context(context), verbose(false), enableOptimizations(true),
          targetDialect("linalg") {}

    mlir::LogicalResult importFromFile(const std::string &modelPath, 
                                      mlir::OwningOpRef<mlir::ModuleOp> &module) {
        if (endsWith(modelPath, ".pt") || endsWith(modelPath, ".pth")) {
            return importPyTorchModel(modelPath, module);
        } else if (endsWith(modelPath, ".onnx")) {
            return importFromONNX(modelPath, module);
        } else {
            llvm::errs() << "Unsupported model file format: " << modelPath << "\n";
            return mlir::failure();
        }
    }

    mlir::LogicalResult importFromONNX(const std::string &onnxPath, 
                                      mlir::OwningOpRef<mlir::ModuleOp> &module) {
        if (verbose) {
            llvm::outs() << "Importing ONNX model: " << onnxPath << "\n";
        }
        
        std::ifstream file(onnxPath);
        if (!file.good()) {
            llvm::errs() << "Cannot open ONNX file: " << onnxPath << "\n";
            return mlir::failure();
        }
        file.close();
        
        module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
        
        if (verbose) {
            llvm::outs() << "Successfully created MLIR module from ONNX\n";
        }
        
        return mlir::success();
    }

    void setVerbose(bool v) { verbose = v; }
    void setEnableOptimizations(bool e) { enableOptimizations = e; }
    void setTargetDialect(const std::string &d) { targetDialect = d; }

private:
    mlir::MLIRContext *context;
    bool verbose;
    bool enableOptimizations;
    std::string targetDialect;

    static bool endsWith(const std::string& str, const std::string& suffix) {
        if (suffix.size() > str.size()) return false;
        return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
    }

    mlir::LogicalResult importPyTorchModel(const std::string &modelPath, 
                                         mlir::OwningOpRef<mlir::ModuleOp> &module) {
        if (verbose) {
            llvm::outs() << "Importing PyTorch model: " << modelPath << "\n";
        }
        
        std::ifstream file(modelPath);
        if (!file.good()) {
            llvm::errs() << "Cannot open PyTorch file: " << modelPath << "\n";
            return mlir::failure();
        }
        file.close();
        
        module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
        
        if (verbose) {
            llvm::outs() << "Successfully created MLIR module from PyTorch\n";
        }
        
        return mlir::success();
    }
};

TorchImporter::TorchImporter(mlir::MLIRContext *context)
    : impl(std::make_unique<Impl>(context)) {}

TorchImporter::~TorchImporter() = default;

mlir::LogicalResult TorchImporter::importFromFile(const std::string &modelPath, 
                                                mlir::OwningOpRef<mlir::ModuleOp> &module) {
    return impl->importFromFile(modelPath, module);
}

mlir::LogicalResult TorchImporter::importFromONNX(const std::string &onnxPath, 
                                                mlir::OwningOpRef<mlir::ModuleOp> &module) {
    return impl->importFromONNX(onnxPath, module);
}

void TorchImporter::setVerbose(bool verbose) {
    impl->setVerbose(verbose);
}

void TorchImporter::setEnableOptimizations(bool enable) {
    impl->setEnableOptimizations(enable);
}

void TorchImporter::setTargetDialect(const std::string &dialectName) {
    impl->setTargetDialect(dialectName);
}

} // namespace Frontend
} // namespace OneCompiler
