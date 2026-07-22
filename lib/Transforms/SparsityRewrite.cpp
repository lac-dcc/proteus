#include "Transforms/SparsityRewrite.h"

#include "Analysis/SparsityLattice.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace proteus {

struct MatmulSparsityRewritePattern
    : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::MatmulOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Grab the lattice from the IR, the user must run the analysis before
    // transforming the IR
    auto latticeAttr = llvm::dyn_cast_if_present<mlir::ArrayAttr>(
        op->getAttr("proteus.lattice"));
    // Otherwise any transformation is going to fail
    if (!latticeAttr) {
      return mlir::failure();
    }

    auto lattice = SparsityLattice::fromAttr(latticeAttr);

    // In the case where everything is dense on both bitvectors there's no
    // sparsity to optimize with
    if ((*lattice)[0].all() && (*lattice)[1].all()) {
      return mlir::failure();
    }

    auto resultType =
        llvm::cast<mlir::RankedTensorType>(op.getResult(0).getType());

    auto loc = op.getLoc();
    auto elementType = resultType.getElementType();

    // In any case we will need to create an all sparse result where we will
    // insert all the miniMatmulOp slices
    auto constOp = mlir::arith::ConstantOp::create(
        rewriter, loc, elementType, rewriter.getFloatAttr(elementType, 0.0));
    auto emptyOp = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), elementType);
    auto fillOpValue =
        mlir::linalg::FillOp::create(rewriter, loc, {constOp}, {emptyOp})
            .getResult(0);

    // In the case where one of the bitvectors is all zeros, then we can just
    // replace the entire op with a zero fill instead of doing any computations
    if ((*lattice)[0].none() || (*lattice)[1].none()) {
      rewriter.replaceOp(op, fillOpValue);
      return mlir::success();
    }

    // Get the contiguous dense ranges to iterate over
    auto rowRanges = SparsityLattice::getDensityRanges((*lattice)[0]);
    auto colRanges = SparsityLattice::getDensityRanges((*lattice)[1]);

    auto lhs = op.getOperand(0);
    auto rhs = op.getOperand(1);
    int64_t k = llvm::cast<mlir::RankedTensorType>(lhs.getType()).getDimSize(1);

    auto strides = mlir::getAsIndexOpFoldResult(rewriter.getContext(), {1, 1});
    auto *context = rewriter.getContext();

    auto miniMatmulize = [&](mlir::Value &finalOpValue, int64_t rowOffset,
                             int64_t rowSize, int64_t colOffset,
                             int64_t colSize) { // NOLINT
      // Extract out of the lhs, the slice that is required
      // to produce the sub domain for the current row and column range
      auto lhsOffsets = mlir::getAsIndexOpFoldResult(context, {rowOffset, 0});
      auto lhsSizes = mlir::getAsIndexOpFoldResult(context, {rowSize, k});
      auto lhsExtractOp = mlir::tensor::ExtractSliceOp::create(
          rewriter, loc, lhs, lhsOffsets, lhsSizes, strides);

      // Extract out of the rhs, the slice that is required
      // to produce the sub domain for the current row and column range
      auto rhsOffsets = mlir::getAsIndexOpFoldResult(context, {0, colOffset});
      auto rhsSizes = mlir::getAsIndexOpFoldResult(context, {k, colSize});
      auto rhsExtractOp = mlir::tensor::ExtractSliceOp::create(
          rewriter, loc, rhs, rhsOffsets, rhsSizes, strides);

      // Extract out of the fill, the zero slice that is required
      // to produce the sub domain for the current row and column range
      auto initOffsets =
          mlir::getAsIndexOpFoldResult(context, {rowOffset, colOffset});
      auto initSizes =
          mlir::getAsIndexOpFoldResult(context, {rowSize, colSize});
      auto initSlice = mlir::tensor::ExtractSliceOp::create(
          rewriter, loc, fillOpValue, initOffsets, initSizes, strides);

      // Do the mini matmul for the subdomain of the current row and column
      // ranges
      auto miniMatmulOp = mlir::linalg::MatmulOp::create(
          rewriter, loc, initSlice.getType(), {lhsExtractOp, rhsExtractOp},
          {initSlice});

      // Insert the mini matmul as a slice to the final rewrite result
      auto finalOffsets =
          mlir::getAsIndexOpFoldResult(context, {rowOffset, colOffset});
      auto finalSizes =
          mlir::getAsIndexOpFoldResult(context, {rowSize, colSize});
      finalOpValue = mlir::tensor::InsertSliceOp::create(
          rewriter, loc, miniMatmulOp->getResult(0), finalOpValue, finalOffsets,
          finalSizes, strides);
    };

    auto finalOpValue = fillOpValue;
    for (auto [rowOffset, rowSize] : rowRanges) {
      for (auto [colOffset, colSize] : colRanges) {
        miniMatmulize(finalOpValue, rowOffset, rowSize, colOffset, colSize);
      }
    }

    rewriter.replaceOp(op, finalOpValue);
    return mlir::success();
  }
};

void SparsityRewritePass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<MatmulSparsityRewritePattern>(patterns.getContext());

  if (mlir::failed(
          mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createSparsityRewritePass() {
  return std::make_unique<SparsityRewritePass>();
}

} // namespace proteus
