#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace proteus {

/**
 * @brief Rewrites a `linalg.pooling_nchw_max` annotated with a
 * `proteus.lattice` attribute into `scf` operations.
 * */
struct PoolMaxNchwSparsityScfRewritePattern
    : public mlir::OpRewritePattern<mlir::linalg::PoolingNchwMaxOp> {
  PoolMaxNchwSparsityScfRewritePattern(mlir::MLIRContext *context,
                                       unsigned &numRewrites)
      : OpRewritePattern(context), numRewrites(numRewrites) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::PoolingNchwMaxOp op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  unsigned &numRewrites; // NOLINT
};

} // namespace proteus
