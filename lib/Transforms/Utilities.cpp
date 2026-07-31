#include "Transforms/Utilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

mlir::Value
proteus::checkIterDensity(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value iter,
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
