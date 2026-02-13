#include "OneCompiler/OneCompiler.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>
#include <memory>

using namespace llvm;

static cl::OptionCategory OneCompilerCategory("OneCompiler options");

static cl::opt<std::string> InputFile(
    cl::Positional, cl::desc("<input file>"),
    cl::Required, cl::cat(OneCompilerCategory));

static cl::opt<std::string> OutputFile(
    "o", cl::desc("Output file"), cl::init(""),
    cl::cat(OneCompilerCategory));

static cl::opt<bool> Verbose(
    "v", cl::desc("Enable verbose output"), cl::init(false),
    cl::cat(OneCompilerCategory));

static cl::opt<std::string> Target(
    "target", cl::desc("Target dialect (linalg, tosa, stablehlo)"), 
    cl::init("linalg"), cl::cat(OneCompilerCategory));

static cl::opt<bool> EmitMLIR(
    "emit-mlir", cl::desc("Emit MLIR IR to output"), cl::init(false),
    cl::cat(OneCompilerCategory));

static cl::opt<bool> EmitLLVM(
    "emit-llvm", cl::desc("Emit LLVM IR to output"), cl::init(false),
    cl::cat(OneCompilerCategory));

static cl::opt<std::string> InputFormat(
    "input-format", cl::desc("Input format (torch, onnx, mlir)"),
    cl::init(""), cl::cat(OneCompilerCategory));

static void printVersion(raw_ostream &os) {
    os << "OneCompiler version 0.0.1\n";
    os << "An MLIR-based AI compiler\n";
}

static bool endsWith(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

int main(int argc, char **argv) {
    InitLLVM y(argc, argv);
    
    cl::SetVersionPrinter(printVersion);
    cl::ParseCommandLineOptions(argc, argv, "OneCompiler - MLIR-based AI Compiler\n");
    
    if (Verbose) {
        outs() << "OneCompiler v0.0.1\n";
        outs() << "Input: " << InputFile << "\n";
        if (!OutputFile.empty()) {
            outs() << "Output: " << OutputFile << "\n";
        }
        outs() << "Target: " << Target << "\n";
    }
    
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    
    std::string actualOutputFile = OutputFile;
    if (actualOutputFile.empty()) {
        actualOutputFile = "output.mlir";
    }
    
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::LogicalResult result = mlir::failure();
    
    std::string inputFile = InputFile;
    
    if (endsWith(inputFile, ".mlir") || InputFormat == "mlir") {
        if (Verbose) {
            outs() << "Parsing MLIR file...\n";
        }
        
        auto sourceMgr = std::make_shared<llvm::SourceMgr>();
        auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFile);
        if (std::error_code ec = fileOrErr.getError()) {
            errs() << "Error: Cannot open file: " << inputFile << "\n";
            return 1;
        }
        
        sourceMgr->AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
        
        mlir::ParserConfig parserConfig(&context);
        module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, parserConfig);
        if (!module) {
            errs() << "Error: Failed to parse MLIR file: " << inputFile << "\n";
            return 1;
        }
        
        result = mlir::success();
        
        if (Verbose) {
            outs() << "MLIR parsed successfully!\n";
        }
    } else if (endsWith(inputFile, ".pt") || endsWith(inputFile, ".pth") || 
        InputFormat == "torch") {
        if (Verbose) {
            outs() << "Importing PyTorch model...\n";
        }
        OneCompiler::Compiler compiler;
        result = compiler.importFromTorchModel(inputFile);
    } else if (endsWith(inputFile, ".onnx") || InputFormat == "onnx") {
        if (Verbose) {
            outs() << "Importing ONNX model...\n";
        }
        OneCompiler::Compiler compiler;
        result = compiler.importFromONNXModel(inputFile);
    } else {
        errs() << "Error: Unknown input format for file: " << InputFile << "\n";
        errs() << "Supported formats: .mlir (MLIR), .pt, .pth (PyTorch), .onnx (ONNX)\n";
        errs() << "Or use --input-format=mlir|torch|onnx to specify format\n";
        return 1;
    }
    
    if (failed(result)) {
        errs() << "Error: Failed to compile input file: " << InputFile << "\n";
        return 1;
    }
    
    if (Verbose) {
        outs() << "Emitting MLIR to: " << actualOutputFile << "\n";
    }
    
    if (actualOutputFile == "-") {
        if (module) {
            module->print(outs());
            outs() << "\n";
        }
    } else {
        std::error_code ec;
        llvm::raw_fd_ostream outFile(actualOutputFile, ec, llvm::sys::fs::OF_None);
        if (ec) {
            errs() << "Error: Cannot open output file: " << actualOutputFile << "\n";
            return 1;
        }
        
        if (module) {
            module->print(outFile);
            outFile << "\n";
        }
    }
    
    if (Verbose) {
        outs() << "Compilation successful!\n";
        if (actualOutputFile != "-") {
            outs() << "Output written to: " << actualOutputFile << "\n";
        }
    }
    
    return 0;
}
