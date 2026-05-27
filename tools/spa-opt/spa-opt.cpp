#include "SPAPass.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();

    mlir::PassRegistration<spa::SPAPass>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tensor Slice Sparsity Propagation Pass\n", registry));
}
