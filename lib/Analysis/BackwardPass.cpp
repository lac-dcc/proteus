#include "Analysis/BackwardPass.h"

#include "Analysis/SparsityEngine.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

void proteus::BackwardPass::run(mlir::Block *block, SparsityEngine &analysis) {
  // This time we are going backwards on the block, hence llvm::reverse
  for (auto &op : llvm::reverse(block->getOperations())) {
    analysis.visit<BackwardPass>(op);
  }
}

void proteus::BackwardPass::visit(mlir::Operation &op,
                                  SparsityEngine &analysis) {
  if (op.getNumResults() != 1) {
    return;
  }

  // Based on the mlir operation we dispatch the appropriate transfer function
  mlir::TypeSwitch<mlir::Operation *>(&op)
      .Case<mlir::linalg::MatmulOp, mlir::linalg::AddOp, mlir::linalg::MatvecOp,
            mlir::linalg::VecmatOp, mlir::linalg::TransposeOp,
            mlir::linalg::BatchMatmulOp, mlir::linalg::FillOp,
            mlir::linalg::BroadcastOp, mlir::linalg::Conv2DOp,
            mlir::linalg::Conv2DNchwFchwOp, mlir::linalg::Conv2DNhwcHwcfOp,
            mlir::linalg::PoolingNchwMaxOp, mlir::linalg::PoolingNchwSumOp,
            mlir::linalg::DepthwiseConv2DNchwChwOp, mlir::tensor::PadOp,
            mlir::tensor::ConcatOp, mlir::tensor::EmptyOp,
            mlir::tensor::ExpandShapeOp, mlir::tensor::ExtractSliceOp,
            mlir::tensor::CollapseShapeOp>(
          [&](auto typedOp) -> void { visitOp(typedOp, analysis); })
      .Case<mlir::linalg::AbsOp, mlir::linalg::CeilOp, mlir::linalg::FloorOp,
            mlir::linalg::NegFOp, mlir::linalg::DivOp,
            mlir::linalg::DivUnsignedOp, mlir::linalg::CopyOp,
            mlir::linalg::TanhOp, mlir::linalg::SquareOp, mlir::linalg::SqrtOp>(
          [&](auto) -> void { visitPassthroughOp(op, analysis); })
      .Default([](auto) {});
}

void proteus::BackwardPass::visitOp(mlir::linalg::MatmulOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::AddOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::MatvecOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::VecmatOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::TransposeOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::BatchMatmulOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::FillOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::BroadcastOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::tensor::PadOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::tensor::ConcatOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::tensor::EmptyOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::Conv2DOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::Conv2DNchwFchwOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::Conv2DNhwcHwcfOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::DepthwiseConv2DNchwChwOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::PoolingNchwMaxOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::linalg::PoolingNchwSumOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::tensor::ExpandShapeOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::tensor::ExtractSliceOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitOp(mlir::tensor::CollapseShapeOp &op,
                                    SparsityEngine &analysis) { /* TODO */ }

void proteus::BackwardPass::visitPassthroughOp(
    mlir::Operation &op, SparsityEngine &analysis) { /* TODO */ }
