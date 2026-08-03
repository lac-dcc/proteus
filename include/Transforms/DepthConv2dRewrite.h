#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace proteus {

/**
 * @brief Rewrites a `linalg.depthwise_conv_2d_nchw_chw` annotated with a
 * `proteus.lattice` attribute into `scf` operations.
 * */
struct DepthConv2dSparsityScfRewritePattern
    : public mlir::OpRewritePattern<mlir::linalg::DepthwiseConv2DNchwChwOp> {
  DepthConv2dSparsityScfRewritePattern(mlir::MLIRContext *context,
                                       unsigned &numRewrites)
      : OpRewritePattern(context), numRewrites(numRewrites) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::DepthwiseConv2DNchwChwOp op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  unsigned &numRewrites; // NOLINT
};

} // namespace proteus
