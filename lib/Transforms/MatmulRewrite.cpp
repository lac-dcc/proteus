#include "Transforms/MatmulRewrite.h"

#include "Analysis/SparsityLattice.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"

namespace proteus {

/* --------------------------- Linalg Rewrites --------------------------- */

mlir::LogicalResult MatmulSparsityLinalgRewritePattern::matchAndRewrite(
    mlir::linalg::MatmulOp op, mlir::PatternRewriter &rewriter) const {
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

  auto loc = op.getLoc();
  auto Cinit = op.getOutputs()[0]; // NOLINT

  // In the case where one of the bitvectors is all zeros, then we can just
  // replace the entire op with a zero fill instead of doing any computations
  if ((*lattice)[0].none() || (*lattice)[1].none()) {
    rewriter.replaceOp(op, Cinit);
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

  auto miniMatmulize = [&](mlir::Value &Cnew,                  // NOLINT
                           int64_t rowOffset,                  // NOLINT
                           int64_t rowSize, int64_t colOffset, // NOLINT
                           int64_t colSize) {                  // NOLINT
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
    auto initSizes = mlir::getAsIndexOpFoldResult(context, {rowSize, colSize});
    auto initSlice = mlir::tensor::ExtractSliceOp::create(
        rewriter, loc, Cinit, initOffsets, initSizes, strides);

    // Do the mini matmul for the subdomain of the current row and column
    // ranges
    auto miniMatmulOp = mlir::linalg::MatmulOp::create(
        rewriter, loc, initSlice.getType(), {lhsExtractOp, rhsExtractOp},
        {initSlice});

    // Insert the mini matmul as a slice to the final rewrite result
    auto finalOffsets =
        mlir::getAsIndexOpFoldResult(context, {rowOffset, colOffset});
    auto finalSizes = mlir::getAsIndexOpFoldResult(context, {rowSize, colSize});
    Cnew = mlir::tensor::InsertSliceOp::create(
        rewriter, loc, miniMatmulOp->getResult(0), Cnew, finalOffsets,
        finalSizes, strides);
  };

  auto Cnew = Cinit; // NOLINT
  for (auto [rowOffset, rowSize] : rowRanges) {
    for (auto [colOffset, colSize] : colRanges) {
      miniMatmulize(Cnew, rowOffset, rowSize, colOffset, colSize);
    }
  }

  rewriter.replaceOp(op, Cnew);
  return mlir::success();
}

/* ----------------------------- SCF Rewrites ----------------------------- */

mlir::Value MatmulSparsityScfRewritePattern::checkIterDensity(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter,
    llvm::ArrayRef<std::pair<int64_t, int64_t>> ranges) {
  mlir::Value isDense;

  for (auto [offset, size] : ranges) {
    auto lb = mlir::arith::ConstantIndexOp::create(builder, loc, offset);
    auto ub = mlir::arith::ConstantIndexOp::create(builder, loc, offset + size);

    auto geLb = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::sge, iter, lb);
    auto ltUb = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::slt, iter, ub);

    auto inThisRange = mlir::arith::AndIOp::create(builder, loc, geLb, ltUb);
    isDense =
        isDense ? mlir::arith::OrIOp::create(builder, loc, isDense, inThisRange)
                      .getResult()
                : inThisRange.getResult();
  }

  return isDense;
}

mlir::LogicalResult MatmulSparsityScfRewritePattern::matchAndRewrite(
    mlir::linalg::MatmulOp op, mlir::PatternRewriter &rewriter) const {
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
  auto Cinit = op.getOutputs()[0]; // NOLINT

  // In the case where one of the bitvectors is all zeros, then we can just
  // replace the entire op with a zero fill instead of doing any computations
  if ((*lattice)[0].none() || (*lattice)[1].none()) {
    rewriter.replaceOp(op, Cinit);
    return mlir::success();
  }

  // Get the contiguous dense ranges; row i / col j is dense iff it falls in
  // one of these
  auto rowRanges = SparsityLattice::getDensityRanges((*lattice)[0]);
  auto colRanges = SparsityLattice::getDensityRanges((*lattice)[1]);

  auto lhs = op.getOperand(0);
  auto rhs = op.getOperand(1);
  auto lhsType = llvm::cast<mlir::RankedTensorType>(lhs.getType());

  int64_t n = resultType.getDimSize(0);
  int64_t m = resultType.getDimSize(1);
  int64_t k = lhsType.getDimSize(1);

  // Lower bound constant for the loops, they all start from zero
  auto c0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
  // Stride of the loop
  auto c1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
  // Upper bound for iLoop
  auto N = mlir::arith::ConstantIndexOp::create(rewriter, loc, n); // NOLINT
  // Upper bound for jLoop
  auto M = mlir::arith::ConstantIndexOp::create(rewriter, loc, m); // NOLINT
  // Upper bound for kLoop
  auto K = mlir::arith::ConstantIndexOp::create(rewriter, loc, k); // NOLINT

  // -- i loop ---------------------------------------------------------------
  auto iLoop = mlir::scf::ForOp::create(
      rewriter, loc, c0, N, c1, Cinit,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value i,
          mlir::ValueRange iArgs) {
        auto isRowDense = checkIterDensity(b, loc, i, rowRanges);

        // -- j loop ---------------------------------------------------------
        auto jLoop = mlir::scf::ForOp::create(
            b, loc, c0, M, c1, iArgs[0],
            [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value j,
                mlir::ValueRange jArgs) {
              auto Cij = jArgs[0]; // NOLINT
              auto isColDense = checkIterDensity(b, loc, j, colRanges);
              auto cond =
                  mlir::arith::AndIOp::create(b, loc, isRowDense, isColDense);

              // If C[i,j] is dense on both rows and columns
              // -- if statement ---------------------------------------------
              auto ifOp = mlir::scf::IfOp::create(
                  b, loc, cond,

                  // -- then case --------------------------------------------
                  [&](mlir::OpBuilder &b, mlir::Location loc) {
                    // -- k loop ---------------------------------------------
                    auto initAcc = mlir::tensor::ExtractOp::create(
                        b, loc, Cij, mlir::ValueRange{i, j});

                    auto kLoop = mlir::scf::ForOp::create(
                        b, loc, c0, K, c1, initAcc.getResult(),
                        [&](mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value k, mlir::ValueRange kArgs) {
                          auto cij = kArgs[0];
                          auto aik = mlir::tensor::ExtractOp::create(
                              b, loc, lhs, {i, k});
                          auto bkj = mlir::tensor::ExtractOp::create(
                              b, loc, rhs, {k, j});
                          auto mul = mlir::arith::MulFOp::create(
                              b, loc, aik, bkj); // A[i,k] * B[k,j]
                          auto acc = mlir::arith::AddFOp::create(
                              b, loc, cij, mul); // C[i,j] += A[i,k] * B[k,j]
                          mlir::scf::YieldOp::create(b, loc, acc.getResult());
                        });

                    auto cnew = mlir::tensor::InsertOp::create(
                        b, loc, kLoop.getResult(0), Cij, {i, j});
                    mlir::scf::YieldOp::create(b, loc, cnew.getResult());
                  },

                  // -- else case --------------------------------------------
                  [&](mlir::OpBuilder &b, mlir::Location loc) {
                    // Nothing happens here just yield the existing Cij
                    mlir::scf::YieldOp::create(b, loc, Cij);
                  });

              mlir::scf::YieldOp::create(b, loc, ifOp.getResult(0));
            });

        mlir::scf::YieldOp::create(b, loc, jLoop.getResult(0));
      });

  rewriter.replaceOp(op, iLoop.getResult(0));
  return mlir::success();
}

} // namespace proteus
