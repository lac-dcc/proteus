#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace proteus {

/**
 * @brief Rewrites a `linalg.conv_2d_nchw_fchw` annotated with a
 * `proteus.lattice` attribute into mutliple `linalg.conv_2d_nchw_fchw`
 * operations. This is done extracting slices from operands and inserting
 * subdomain convolutions into the accumulator result
 * */
struct Conv2dSparsityLinalgRewritePattern
    : public mlir::OpRewritePattern<mlir::linalg::Conv2DNchwFchwOp> {
  Conv2dSparsityLinalgRewritePattern(mlir::MLIRContext *context,
                                     unsigned &numRewrites)
      : OpRewritePattern(context), numRewrites(numRewrites) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::Conv2DNchwFchwOp op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  unsigned &numRewrites; // NOLINT
};

/**
 * @brief Rewrites a `linalg.conv_2d_nchw_fchw` annotated with a
 * `proteus.lattice` attribute into `scf` operations.
 * */
struct Conv2dSparsityScfRewritePattern
    : public mlir::OpRewritePattern<mlir::linalg::Conv2DNchwFchwOp> {
  Conv2dSparsityScfRewritePattern(mlir::MLIRContext *context,
                                  unsigned &numRewrites)
      : OpRewritePattern(context), numRewrites(numRewrites) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::Conv2DNchwFchwOp op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  unsigned &numRewrites; // NOLINT
};

} // namespace proteus
