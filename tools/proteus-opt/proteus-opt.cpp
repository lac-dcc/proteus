#include "Analysis/SparsityPropagationAnalysis.h"

#include "Dialect/Probe/IR/Probe.h"
#include "Dialect/Probe/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
    mlir::registerAllPasses();

    mlir::DialectRegistry registry;
    registry.insert<mlir::probe::ProbeDialect>();
    mlir::registerAllDialects(registry);

    mlir::PassRegistration<proteus::SparsityPropagationAnalysis>();
    mlir::probe::registerProbePasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tensor Slice Sparsity Propagation Pass\n", registry));
}
