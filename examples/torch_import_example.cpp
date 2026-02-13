#include "OneCompiler/OneCompiler.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::OptionCategory TorchImportExampleCategory("torch-import-example options");
static cl::opt<std::string> InputModel(
    cl::Positional, cl::desc("<input PyTorch model file (.pt, .pth) or ONNX file (.onnx)>"),
    cl::Required, cl::cat(TorchImportExampleCategory));
static cl::opt<std::string> OutputFile(
    "o", cl::desc("Output MLIR file"), cl::init("output.mlir"),
    cl::cat(TorchImportExampleCategory));
static cl::opt<bool> Verbose(
    "v", cl::desc("Enable verbose output"), cl::init(false),
    cl::cat(TorchImportExampleCategory));

int main(int argc, char **argv) {
    InitLLVM y(argc, argv);
    cl::ParseCommandLineOptions(argc, argv, "OneCompiler Torch Import Example\n");
    
    OneCompiler::Compiler compiler;
    
    if (compiler.importFromTorchModel(InputModel).failed()) {
        errs() << "Failed to import PyTorch model: " << InputModel << "\n";
        return 1;
    }
    
    outs() << "Successfully imported model: " << InputModel << "\n";
    outs() << "MLIR module written to: " << OutputFile << "\n";
    
    return 0;
}
