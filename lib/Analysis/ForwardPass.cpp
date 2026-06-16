#include "Analysis/ForwardPass.h"

#include "Analysis/SparsityEngine.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

Result proteus::ForwardPass::run(mlir::Block *block, SparsityEngine &analysis) {
  for (auto &op : block->getOperations()) {
    if (analysis.visit<ForwardPass>(op).failed()) {
      return op.emitError("Proteus failed to properly visit op:")
             << op.getName();
    }
  }

  return mlir::success();
}

Result proteus::ForwardPass::visit(mlir::Operation &op,
                                   SparsityEngine &analysis) {
  if (op.getNumResults() != 1) {
    return mlir::success();
  }

  std::optional<SparsityLattice> lattice;

  if (op.hasAttr("proteus.lattice")) {
    auto attr = op.getAttr("proteus.lattice");
    auto arrayAttr = llvm::cast<mlir::ArrayAttr>(attr);
    lattice = SparsityLattice::fromAttr(arrayAttr);
  } else {
    lattice = SparsityLattice::defaultFromValue(op.getResult(0));
  }

  if (lattice.has_value()) {
    analysis.getState().try_emplace(op.getResult(0), lattice.value());

    Result switchResult =
        mlir::TypeSwitch<mlir::Operation *, Result>(&op)
            .Case<mlir::linalg::MatmulOp, mlir::linalg::AddOp,
                  mlir::linalg::MatvecOp, mlir::linalg::VecmatOp>(
                [&](auto typedOp) -> Result {
                  if (visitOp(typedOp, analysis).failed()) {
                    return op.emitError("Failed when visiting op: ")
                           << op.getName();
                  }
                  return mlir::success();
                })
            .Case<mlir::linalg::AbsOp, mlir::linalg::CeilOp,
                  mlir::linalg::FloorOp, mlir::linalg::NegFOp,
                  mlir::linalg::DivOp, mlir::linalg::DivUnsignedOp,
                  mlir::linalg::CopyOp, mlir::linalg::TanhOp,
                  mlir::linalg::SquareOp, mlir::linalg::SqrtOp>(
                [&](auto) -> Result {
                  return visitPassthroughOp(op, analysis);
                })
            .Default([](auto) { return mlir::success(); });
    if (switchResult.failed()) {
      return switchResult;
    }
  }

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::MatmulOp &op,
                                     SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));
  auto *res = analysis.getState(op.getResult(0));

  // We expect that all lattices are contained within the lattice map at this
  // point, if not, we probably have not implemented a tranfer function that
  // could yield results into a matmul op
  if ((lhs == nullptr) || (rhs == nullptr) || (res == nullptr)) {
    return op.emitError("The lattices are not propagated"
                        "properly in op: ")
           << mlir::linalg::MatmulOp::getOperationName();
  }

  // We want to keep any sparsity already in the result, this could happen
  // if an attribute is set on a matmul operation
  (*res)[0] &= (*lhs)[0];
  (*res)[1] &= (*rhs)[1];

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::AddOp &op,
                                     SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));
  auto *res = analysis.getState(op.getResult(0));

  // We expect that all lattices are contained within the lattice map at this
  // point, if not, we probably have not implemented a tranfer function that
  // could yield results into an add op
  if ((lhs == nullptr) || (rhs == nullptr) || (res == nullptr)) {
    return op.emitError("The lattices are not propagated"
                        "properly in op: ")
           << mlir::linalg::AddOp::getOperationName();
  }

  for (std::size_t i = 0; i < res->rank(); i++) {
    // llvm::BitVector does not support binary operations such as lhs & rhs
    // so we have to go with a unary operation instead here.
    llvm::BitVector temp = (*lhs)[i];
    temp |= (*rhs)[i];
    // Again we need to respect any preexisting lattices
    (*res)[i] = temp;
  }

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::MatvecOp &op,
                                     SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op->getOperand(0));
  auto *rhs = analysis.getState(op->getOperand(1));
  auto *res = analysis.getState(op->getResult(0));

  if ((lhs == nullptr) || (rhs == nullptr) || (res == nullptr)) {
    return op.emitError("The lattices are not propagated properly in op: ")
           << mlir::linalg::MatvecOp::getOperationName();
  }

  if ((*rhs)[0].none()) {
    (*res)[0].reset();
  } else {
    (*res)[0] = (*lhs)[0];
  }

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::VecmatOp &op,
                                     SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op->getOperand(0));
  auto *rhs = analysis.getState(op->getOperand(1));
  auto *res = analysis.getState(op->getResult(0));

  if ((lhs == nullptr) || (rhs == nullptr) || (res == nullptr)) {
    return op.emitError("The lattices are not propagated properly in op: ")
           << mlir::linalg::VecmatOp::getOperationName();
  }

  if ((*lhs)[0].none()) {
    (*res)[0].reset();
  } else {
    (*res)[0] = (*rhs)[1];
  }

  return mlir::success();
}

Result proteus::ForwardPass::visitPassthroughOp(mlir::Operation &op,
                                                SparsityEngine &analysis) {
  auto *operand = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op.getOpResult(0));

  if ((operand == nullptr) || (res == nullptr)) {
    return op.emitError("The lattices are not propagated properly in op: ")
           << op.getName();
  }

  for (std::size_t i = 0; i < res->rank(); i++) {
    (*res)[i] = (*operand)[i];
  }

  return mlir::success();
}
