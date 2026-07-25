#include "Transforms/Conv2dNchwFchwRewrite.h"

#include "Analysis/SparsityLattice.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"

namespace proteus {

/* --------------------------- Linalg Rewrites --------------------------- */

mlir::LogicalResult Conv2dSparsityRewritePattern::matchAndRewrite(
    mlir::linalg::Conv2DNchwFchwOp op, mlir::PatternRewriter &rewriter) const {
  // Grab the lattice from the IR, the user must run the analysis before
  // transforming the IR
  auto latticeAttr = llvm::dyn_cast_if_present<mlir::ArrayAttr>(
      op->getAttr("proteus.lattice"));
  // Otherwise any transformation is going to fail
  if (!latticeAttr) {
    return mlir::failure();
  }

  auto lattice = SparsityLattice::fromAttr(latticeAttr);

  // In the case where everything is dense there's no sparsity to optimize with
  if ((*lattice)[0].all() && (*lattice)[1].all() && (*lattice)[2].all() &&
      (*lattice)[3].all()) {
    return mlir::failure();
  }

  auto loc = op.getLoc();
  auto Cinit = op.getOutputs()[0]; // NOLINT

  // In the case where one of the bitvectors is all zeros, then we can just
  // replace the entire op with a zero accumulator instead of doing any
  // computations
  if ((*lattice)[0].none() || (*lattice)[1].none() || (*lattice)[2].none() ||
      (*lattice)[3].none()) {
    rewriter.replaceOp(op, Cinit);
    return mlir::success();
  }

  // Get the contiguous dense ranges to iterate over
  auto batchRanges = SparsityLattice::getDensityRanges((*lattice)[0]);
  auto fRanges = SparsityLattice::getDensityRanges((*lattice)[1]);
  auto rowRanges = SparsityLattice::getDensityRanges((*lattice)[2]);
  auto colRanges = SparsityLattice::getDensityRanges((*lattice)[3]);

  auto input = op.getOperand(0);
  auto filter = op.getOperand(1);
  auto filterType = llvm::cast<mlir::RankedTensorType>(filter.getType());

  // Convolution parameters
  int64_t c = filterType.getDimSize(1);
  int64_t kh = filterType.getDimSize(2);
  int64_t kw = filterType.getDimSize(3);
  auto strideH = op.getStrides().getValues<int64_t>()[0];
  auto strideW = op.getStrides().getValues<int64_t>()[1];
  auto dilationH = op.getDilations().getValues<int64_t>()[0];
  auto dilationW = op.getDilations().getValues<int64_t>()[1];

  // These strides are for the insertion and extraction of slices, not related
  // to the convolution ones
  auto sliceStrides =
      mlir::getAsIndexOpFoldResult(rewriter.getContext(), {1, 1, 1, 1});
  auto *context = rewriter.getContext();

  auto miniConvolize = [&](mlir::Value &Cnew,                  // NOLINT
                           int64_t batchOffset,                // NOLINT
                           int64_t batchSize, int64_t fOffset, // NOLINT
                           int64_t fSize,                      // NOLINT
                           int64_t rowOffset,                  // NOLINT
                           int64_t rowSize, int64_t colOffset, // NOLINT
                           int64_t colSize) {                  // NOLINT
    int64_t inRowOffset = rowOffset * strideH;
    int64_t inColOffset = colOffset * strideW;
    // https://docs.pytorch.org/docs/2.13/generated/torch.nn.Conv2d.html solved
    // for Hin, Win
    int64_t inRowSize = ((rowSize - 1) * strideH) + ((kh - 1) * dilationH) + 1;
    int64_t inColSize = ((colSize - 1) * strideW) + ((kw - 1) * dilationW) + 1;

    // Extract out of the input, the input window that slides over to produce
    // the sub domain for the current row and column range
    auto inputOffsets = mlir::getAsIndexOpFoldResult(
        context, {batchOffset, 0, inRowOffset, inColOffset});
    auto inputSizes = mlir::getAsIndexOpFoldResult(
        context, {batchSize, c, inRowSize, inColSize});
    auto inputExtractOp = mlir::tensor::ExtractSliceOp::create(
        rewriter, loc, input, inputOffsets, inputSizes, sliceStrides);

    // Extract out of the filter, the filter slice for the current output
    // channel range
    auto filterOffsets =
        mlir::getAsIndexOpFoldResult(context, {fOffset, 0, 0, 0});
    auto filterSizes =
        mlir::getAsIndexOpFoldResult(context, {fSize, c, kh, kw});
    auto filterExtractOp = mlir::tensor::ExtractSliceOp::create(
        rewriter, loc, filter, filterOffsets, filterSizes, sliceStrides);

    // Extract out of the accumulator, the zero slice that is required to
    // produce the sub domain for the current batch, channel, row and column
    // range
    auto initOffsets = mlir::getAsIndexOpFoldResult(
        context, {batchOffset, fOffset, rowOffset, colOffset});
    auto initSizes = mlir::getAsIndexOpFoldResult(
        context, {batchSize, fSize, rowSize, colSize});
    auto initSlice = mlir::tensor::ExtractSliceOp::create(
        rewriter, loc, Cinit, initOffsets, initSizes, sliceStrides);

    // Do the mini conv for the subdomain of the current ranges, preserving the
    // original op's sliceStrides & dilations
    auto miniConvOp = mlir::linalg::Conv2DNchwFchwOp::create(
        rewriter, loc, initSlice.getType(),
        mlir::ValueRange{inputExtractOp, filterExtractOp},
        mlir::ValueRange{initSlice}, op.getStrides(), op.getDilations());

    // Insert the mini conv as a slice to the final rewrite result
    Cnew = mlir::tensor::InsertSliceOp::create(
        rewriter, loc, miniConvOp->getResult(0), Cnew, initOffsets, initSizes,
        sliceStrides);
  };

  auto Cnew = Cinit; // NOLINT
  for (auto [batchOffset, batchSize] : batchRanges) {
    for (auto [fOffset, fSize] : fRanges) {
      for (auto [rowOffset, rowSize] : rowRanges) {
        for (auto [colOffset, colSize] : colRanges) {
          miniConvolize(Cnew, batchOffset, batchSize, fOffset, fSize, rowOffset,
                        rowSize, colOffset, colSize);
        }
      }
    }
  }

  rewriter.replaceOp(op, Cnew);
  return mlir::success();
}

} // namespace proteus
