#include "Analysis/LateralPass.h"

#include "Analysis/SparsityEngine.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

void proteus::LateralPass::run(mlir::Block *block, SparsityEngine &analysis) {
  for (auto &op : block->getOperations()) {
    analysis.visit<LateralPass>(op);
  }
}

void proteus::LateralPass::visit(mlir::Operation &op,
                                 SparsityEngine &analysis) {
  // Based on the mlir operation we dispatch the appropriate transfer function
  mlir::TypeSwitch<mlir::Operation *>(&op)
      .Case<mlir::linalg::MatmulOp, mlir::linalg::AddOp, mlir::linalg::MatvecOp,
            mlir::linalg::VecmatOp, mlir::linalg::BatchMatmulOp>(
          [&](auto typedOp) -> void { visitOp(typedOp, analysis); })
      .Default([](auto) {});
}

void proteus::LateralPass::visitOp(mlir::linalg::AddOp &op,
                                   SparsityEngine &analysis) { /* TODO */ }

void proteus::LateralPass::visitOp(mlir::linalg::MatmulOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));

  (*lhs)[1] &= (*rhs)[0];
  (*rhs)[0] &= (*lhs)[1];
}

void proteus::LateralPass::visitOp(mlir::linalg::MatvecOp &op,
                                   SparsityEngine &analysis) { /* TODO */ }

void proteus::LateralPass::visitOp(mlir::linalg::VecmatOp &op,
                                   SparsityEngine &analysis) { /* TODO */ }

void proteus::LateralPass::visitOp(mlir::linalg::BatchMatmulOp &op,
                                   SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));

  (*lhs)[2] &= (*rhs)[1];
  (*rhs)[1] &= (*lhs)[2];
}
