#include "Transforms/Conv2dNchwFchwRewrite.h"

#include "Analysis/SparsityLattice.h"
#include "Transforms/Utilities.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace proteus {

/* --------------------------- Linalg Rewrites --------------------------- */

LogicalResult Conv2dSparsityLinalgRewritePattern::matchAndRewrite(
    linalg::Conv2DNchwFchwOp op, mlir::PatternRewriter &rewriter) const {
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
  if (lattice[0].all() && lattice[1].all() && lattice[2].all() &&
      lattice[3].all()) {
    return mlir::failure();
  }

  auto loc = op.getLoc();
  auto Cinit = op.getOutputs()[0]; // NOLINT

  // In the case where one of the bitvectors is all zeros, then we can just
  // replace the entire op with a zero accumulator instead of doing any
  // computations
  if (lattice[0].none() || lattice[1].none() || lattice[2].none() ||
      lattice[3].none()) {
    ++numRewrites;
    rewriter.replaceOp(op, Cinit);
    return mlir::success();
  }

  // Get the contiguous dense ranges to iterate over
  auto batchRanges = SparsityLattice::getDensityRanges(lattice[0]);
  auto fRanges = SparsityLattice::getDensityRanges(lattice[1]);
  auto rowRanges = SparsityLattice::getDensityRanges(lattice[2]);
  auto colRanges = SparsityLattice::getDensityRanges(lattice[3]);

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
      getAsIndexOpFoldResult(rewriter.getContext(), {1, 1, 1, 1});
  auto *context = rewriter.getContext();

  auto miniConvolize = [&](Value &Cnew,                        // NOLINT
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
    auto filterOffsets = getAsIndexOpFoldResult(context, {fOffset, 0, 0, 0});
    auto filterSizes = getAsIndexOpFoldResult(context, {fSize, c, kh, kw});
    auto filterExtractOp = tensor::ExtractSliceOp::create(
        rewriter, loc, filter, filterOffsets, filterSizes, sliceStrides);

    // Extract out of the accumulator, the zero slice that is required to
    // produce the sub domain for the current batch, channel, row and column
    // range
    auto initOffsets = getAsIndexOpFoldResult(
        context, {batchOffset, fOffset, rowOffset, colOffset});
    auto initSizes =
        getAsIndexOpFoldResult(context, {batchSize, fSize, rowSize, colSize});
    auto initSlice = tensor::ExtractSliceOp::create(
        rewriter, loc, Cinit, initOffsets, initSizes, sliceStrides);

    // Do the mini conv for the subdomain of the current ranges, preserving the
    // original op's sliceStrides & dilations
    auto miniConvOp = linalg::Conv2DNchwFchwOp::create(
        rewriter, loc, initSlice.getType(),
        ValueRange{inputExtractOp, filterExtractOp}, ValueRange{initSlice},
        op.getStrides(), op.getDilations());

    // Insert the mini conv as a slice to the final rewrite result
    Cnew = tensor::InsertSliceOp::create(rewriter, loc,
                                         miniConvOp->getResult(0), Cnew,
                                         initOffsets, initSizes, sliceStrides);
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

  ++numRewrites;
  rewriter.replaceOp(op, Cnew);
  return success();
}

/* ----------------------------- SCF Rewrites ----------------------------- */

LogicalResult Conv2dSparsityScfRewritePattern::matchAndRewrite(
    linalg::Conv2DNchwFchwOp op, PatternRewriter &rewriter) const {
  // Grab the lattice from the IR, the user must run the analysis before
  // transforming the IR
  auto latticeAttr =
      llvm::dyn_cast_if_present<ArrayAttr>(op->getAttr("proteus.lattice"));
  // Otherwise any transformation is going to fail
  if (!latticeAttr) {
    return failure();
  }

  auto lattice = SparsityLattice::fromAttr(latticeAttr);

  // In the case where everything is dense there's no sparsity to optimize with
  if (lattice[0].all() && lattice[1].all() && lattice[2].all() &&
      lattice[3].all()) {
    return failure();
  }

  auto resultType = llvm::cast<RankedTensorType>(op.getResult(0).getType());

  auto loc = op.getLoc();
  auto Cinit = op.getOutputs()[0]; // NOLINT

  // In the case where one of the bitvectors is all zeros, then we can just
  // replace the entire op with a zero accumulator instead of doing any
  // computations
  if (lattice[0].none() || lattice[1].none() || lattice[2].none() ||
      lattice[3].none()) {
    ++numRewrites;
    rewriter.replaceOp(op, Cinit);
    return success();
  }

  // Get the contiguous dense ranges
  auto batchRanges = SparsityLattice::getDensityRanges(lattice[0]);
  auto fRanges = SparsityLattice::getDensityRanges(lattice[1]);
  auto rowRanges = SparsityLattice::getDensityRanges(lattice[2]);
  auto colRanges = SparsityLattice::getDensityRanges(lattice[3]);

  auto input = op.getOperand(0);
  auto filter = op.getOperand(1);
  auto filterType = llvm::cast<RankedTensorType>(filter.getType());

  // -- Convolution parameters -----------------------------------------------
  // Kernel dimension sizes
  int64_t c = filterType.getDimSize(1);
  int64_t kh = filterType.getDimSize(2);
  int64_t kw = filterType.getDimSize(3);
  // Output dimension sizes
  int64_t n = resultType.getDimSize(0);
  int64_t f = resultType.getDimSize(1);
  int64_t oh = resultType.getDimSize(2);
  int64_t ow = resultType.getDimSize(3);
  // Strides & dilations per spatial dimensions
  auto strideH = op.getStrides().getValues<int64_t>()[0];
  auto strideW = op.getStrides().getValues<int64_t>()[1];
  auto dilationH = op.getDilations().getValues<int64_t>()[0];
  auto dilationW = op.getDilations().getValues<int64_t>()[1];

  // -- Convolution Constant Conversions ------------------------------------
  // Strides & dilations per spatial dimensions
  auto constStrideH = arith::ConstantIndexOp::create(rewriter, loc, strideH);
  auto constStrideW = arith::ConstantIndexOp::create(rewriter, loc, strideW);
  auto constDilH = arith::ConstantIndexOp::create(rewriter, loc, dilationH);
  auto constDilW = arith::ConstantIndexOp::create(rewriter, loc, dilationW);
  // Lower bound constant for the loops, they all start at zero
  auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  // Stride of the loop
  auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  // Upper bounds for the output loops
  auto N = arith::ConstantIndexOp::create(rewriter, loc, n);   // NOLINT
  auto F = arith::ConstantIndexOp::create(rewriter, loc, f);   // NOLINT
  auto OH = arith::ConstantIndexOp::create(rewriter, loc, oh); // NOLINT
  auto OW = arith::ConstantIndexOp::create(rewriter, loc, ow); // NOLINT
  // Upper bounds for the kernel loops
  auto C = arith::ConstantIndexOp::create(rewriter, loc, c);   // NOLINT
  auto KH = arith::ConstantIndexOp::create(rewriter, loc, kh); // NOLINT
  auto KW = arith::ConstantIndexOp::create(rewriter, loc, kw); // NOLINT

  auto then = [&](OpBuilder &b, Location loc, Value Cinit, // NOLINT
                  ValueRange thenArgs) {
    auto n = thenArgs[0];
    auto f = thenArgs[1];
    auto oh = thenArgs[2];
    auto ow = thenArgs[3];
    auto Onchw = // NOLINT
        tensor::ExtractOp::create(b, loc, Cinit, {n, f, oh, ow});

    // -- c loop -------------------------------------------------------------
    auto cLoop = scf::ForOp::create(
        b, loc, c0, C, c1, Onchw.getResult(),
        [&](OpBuilder &b, Location loc, Value c, ValueRange cArgs) {
          auto Onchw = cArgs[0]; // NOLINT

          // -- kh loop ------------------------------------------------------
          auto khLoop = scf::ForOp::create(
              b, loc, c0, KH, c1, Onchw,
              [&](OpBuilder &b, Location loc, Value kh, ValueRange khArgs) {
                auto Onchw = khArgs[0]; // NOLINT

                // -- kw loop ------------------------------------------------
                auto kwLoop = scf::ForOp::create(
                    b, loc, c0, KW, c1, Onchw,
                    [&](OpBuilder &b, Location loc, Value kw,
                        ValueRange kwArgs) {
                      auto Onchw = kwArgs[0]; // NOLINT

                      // https://github.com/llvm/llvm-project/blob/main/mlir/python/mlir/dialects/linalg/opdsl/ops/core_named_ops.py
                      // For conv_2d_nchw_fchw, the output Layout is as follows:
                      // O[n, f, oh, ow] +=
                      // I[n, c, oh * SH + kh * DH, ow * SW + kw * DW] *
                      // K[f, c, kh, kw])
                      auto ih = arith::AddIOp::create(
                          b, loc,
                          arith::MulIOp::create(b, loc, oh, constStrideH),
                          arith::MulIOp::create(b, loc, kh, constDilH));
                      auto iw = arith::AddIOp::create(
                          b, loc,
                          arith::MulIOp::create(b, loc, ow, constStrideW),
                          arith::MulIOp::create(b, loc, kw, constDilW));

                      // Extract the appropriate slices from input and filter
                      auto Inchw = tensor::ExtractOp::create( // NOLINT
                          b, loc, input, {n, c, ih, iw});
                      auto Ffchw = tensor::ExtractOp::create( // NOLINT
                          b, loc, filter, {f, c, kh, kw});

                      auto mul = arith::MulFOp::create(b, loc, Inchw, Ffchw);
                      auto acc = arith::AddFOp::create(b, loc, Onchw, mul);

                      scf::YieldOp::create(b, loc, acc.getResult());
                    });

                scf::YieldOp::create(b, loc, kwLoop.getResult(0));
              });

          scf::YieldOp::create(b, loc, khLoop.getResult(0));
        });

    auto cnew = tensor::InsertOp::create(b, loc, cLoop.getResult(0), Cinit,
                                         {n, f, oh, ow});
    scf::YieldOp::create(b, loc, cnew.getResult());
  };

  // -- n loop ---------------------------------------------------------------
  auto nLoop = scf::ForOp::create(
      rewriter, loc, c0, N, c1, Cinit,
      [&](OpBuilder &b, Location loc, Value n, ValueRange nArgs) {
        auto isNDense = checkIterDensity(b, loc, n, batchRanges);
        auto Cinit = nArgs[0]; // NOLINT

        // -- f loop --------------------------------------------------------
        auto chLoop = scf::ForOp::create(
            b, loc, c0, F, c1, Cinit,
            [&](OpBuilder &b, Location loc, Value f, ValueRange chArgs) {
              auto isFDense = checkIterDensity(b, loc, f, fRanges);
              auto Cinit = chArgs[0]; // NOLINT

              // -- oh loop --------------------------------------------------
              auto ohLoop = scf::ForOp::create(
                  b, loc, c0, OH, c1, Cinit,
                  [&](OpBuilder &b, Location loc, Value oh, ValueRange ohArgs) {
                    auto isHDense = checkIterDensity(b, loc, oh, rowRanges);
                    auto Cinit = ohArgs[0]; // NOLINT

                    // -- ow loop --------------------------------------------
                    auto owLoop = scf::ForOp::create(
                        b, loc, c0, OW, c1, Cinit,
                        [&](OpBuilder &b, Location loc, Value ow,
                            ValueRange owArgs) {
                          auto isWDense =
                              checkIterDensity(b, loc, ow, colRanges);
                          auto Cinit = owArgs[0]; // NOLINT

                          auto cond = arith::AndIOp::create(
                              b, loc,
                              arith::AndIOp::create(b, loc, isNDense, isFDense),
                              arith::AndIOp::create(b, loc, isHDense,
                                                    isWDense));

                          // -- if statement ---------------------------------
                          auto ifOp = scf::IfOp::create(
                              b, loc, cond,

                              // -- then case --------------------------------
                              [&](OpBuilder &b, Location loc) {
                                then(b, loc, Cinit, {n, f, oh, ow});
                              },

                              // -- else case --------------------------------
                              [&](OpBuilder &b, Location loc) {
                                // Nothing happens here, just yield
                                scf::YieldOp::create(b, loc, Cinit);
                              });

                          scf::YieldOp::create(b, loc, ifOp.getResult(0));
                        });

                    scf::YieldOp::create(b, loc, owLoop.getResult(0));
                  });

              scf::YieldOp::create(b, loc, ohLoop.getResult(0));
            });

        scf::YieldOp::create(b, loc, chLoop.getResult(0));
      });

  ++numRewrites;
  rewriter.replaceOp(op, nLoop.getResult(0));
  return success();
}

} // namespace proteus
