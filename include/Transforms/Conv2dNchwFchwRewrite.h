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
struct Conv2dSparsityRewritePattern
    : public mlir::OpRewritePattern<mlir::linalg::Conv2DNchwFchwOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::Conv2DNchwFchwOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace proteus
