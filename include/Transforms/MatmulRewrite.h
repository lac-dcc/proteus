#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace proteus {

/**
 * @brief Rewrites a `linalg.matmul` annotated with a `proteus.lattice`
 * attribute into mutliple `linalg.matmul` operations. This is done
 * extracting slices from operands and inserting subdomain matmuls into
 * a zero filled result tensor
 * */
struct MatmulSparsityLinalgRewritePattern
    : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
  MatmulSparsityLinalgRewritePattern(mlir::MLIRContext *context,
                                     unsigned &numRewrites)
      : OpRewritePattern(context), numRewrites(numRewrites) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::MatmulOp op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  unsigned &numRewrites; // NOLINT
};

/**
 * @brief Rewrites a `linalg.matmul` annotated with a `proteus.lattice`
 * attribute into `scf` operations.
 * */
struct MatmulSparsityScfRewritePattern
    : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
  MatmulSparsityScfRewritePattern(mlir::MLIRContext *context,
                                  unsigned &numRewrites)
      : OpRewritePattern(context), numRewrites(numRewrites) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::MatmulOp op,
                  mlir::PatternRewriter &rewriter) const override;

  static mlir::Value
  checkIterDensity(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value iter,
                   llvm::ArrayRef<std::pair<int64_t, int64_t>> ranges);

private:
  unsigned &numRewrites; // NOLINT
};

} // namespace proteus
