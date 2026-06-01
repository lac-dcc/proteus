#include "Analysis/SparsityPropagationAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();

    mlir::PassRegistration<proteus::SparsityPropagationAnalysis>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tensor Slice Sparsity Propagation Pass\n", registry));
}
