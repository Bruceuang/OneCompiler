#ifndef ONECOMPILER_ONECOMPILER_H
#define ONECOMPILER_ONECOMPILER_H

#include "OneCompiler/OneCompiler.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h" 
#include "mlir/Parser/Parser.h" 
#include "mlir/Pass/PassManager.h"

namespace OneCompiler {

class OneCompiler {
public:
    OneCompiler();
    ~OneCompiler();

    // 编译主入口
    mlir::LogicalResult compile(const std::string& inputFile, 
                               const std::string& outputFile);

    // 注册所有方言和passes
    void registerAllDialects();
    void registerAllPasses();

private:
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
};

} // namespace OneCompiler

#endif // ONECOMPILER_ONECOMPILER_H
