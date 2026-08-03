#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace proteus {

/**
 * @brief Rewrites a `linalg.pooling_nchw_sum` annotated with a
 * `proteus.lattice` attribute into `scf` operations.
 * */
struct PoolSumNchwSparsityScfRewritePattern
    : public mlir::OpRewritePattern<mlir::linalg::PoolingNchwSumOp> {
  PoolSumNchwSparsityScfRewritePattern(mlir::MLIRContext *context,
                                       unsigned &numRewrites)
      : OpRewritePattern(context), numRewrites(numRewrites) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::PoolingNchwSumOp op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  unsigned &numRewrites; // NOLINT
};

} // namespace proteus
