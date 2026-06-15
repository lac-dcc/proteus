#include "Analysis/ForwardPass.h"

#include "Analysis/SparsityEngine.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

Result proteus::ForwardPass::run(mlir::Block *block, SparsityEngine &analysis) {
  for (auto &op : block->getOperations()) {
    if (analysis.visit<ForwardPass>(op).failed())
      return op.emitError("Proteus failed to properly visit op:")
             << op.getName();
  }

  return mlir::success();
}

Result proteus::ForwardPass::visit(mlir::Operation &op,
                                   SparsityEngine &analysis) {
  if (op.getNumResults() != 1)
    return mlir::success();

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

    mlir::TypeSwitch<mlir::Operation *>(&op)
        .Case<mlir::linalg::MatmulOp, mlir::linalg::AddOp, mlir::linalg::AbsOp,
              mlir::linalg::CeilOp, mlir::linalg::FloorOp, mlir::linalg::NegfOp,
              mlir::linalg::DivOp, mlir::linalg::DivUnsignedOp
              /*, mlir::linalg::ManyOtherOps */>(
            [&](auto typedOp) { visitOp(typedOp, analysis); })
        .Default([](auto) {});
  }

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::MatmulOp op,
                                     SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));
  auto *res = analysis.getState(op.getResult(0));

  // We expect that all lattices are contained within the lattice map at this
  // point, if not, we probably have not implemented a tranfer function that
  // could yield results into a matmul op
  if (!lhs || !rhs || !res)
    return op.emitError("The lattices are not propagated"
                        "properly in op: ")
           << op.getOperationName();

  // We want to keep any sparsity already in the result, this could happen
  // if an attribute is set on a matmul operation
  (*res)[0] &= (*lhs)[0];
  (*res)[1] &= (*rhs)[1];

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::AddOp op,
                                     SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));
  auto *res = analysis.getState(op.getResult(0));

  // We expect that all lattices are contained within the lattice map at this
  // point, if not, we probably have not implemented a tranfer function that
  // could yield results into an add op
  if (!lhs || !rhs || !res)
    return op.emitError("The lattices are not propagated"
                        "properly in op: ")
           << op.getOperationName();

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

Result proteus::ForwardPass::visitOp(mlir::linalg::AbsOp op,
                                     SparsityEngine &analysis) {
  auto *oper = analysis.getState(op->getOperand(0));
  auto *res = analysis.getState(op->getOpResult(0));

  if (!oper || !res)
    return op.emitError("The lattices are not propagated"
                        "properly in op: ")
           << op.getOperationName();

  for (std::size_t i = 0; i < res->rank(); i++) {
    (*res)[i] = (*oper)[i];
  }

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::CeilOp op,
                                     SparsityEngine &analysis) {
  auto *oper = analysis.getState(op->getOperand(0));
  auto *res = analysis.getState(op->getOpResult(0));

  if (!oper || !res)
    return op.emitError("The lattices are not propagated properly in op: ")
           << op.getOperationName();

  for (std::size_t i = 0; i < res->rank(); i++)
    (*res)[i] = (*oper)[i];

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::FloorOp op,
                                     SparsityEngine &analysis) {
  auto *oper = analysis.getState(op->getOperand(0));
  auto *res = analysis.getState(op->getOpResult(0));

  if (!oper || !res)
    return op.emitError("The lattices are not propagated properly in op: ")
           << op.getOperationName();

  for (std::size_t i = 0; i < res->rank(); i++)
    (*res)[i] = (*oper)[i];

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::NegfOp op,
                                     SparsityEngine &analysis) {
  auto *oper = analysis.getState(op->getOperand(0));
  auto *res = analysis.getState(op->getOpResult(0));

  if (!oper || !res)
    return op.emitError("The lattices are not propagated properly in op: ")
           << op.getOperationName();

  for (std::size_t i = 0; i < res->rank(); i++)
    (*res)[i] = (*oper)[i];

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::DivOp op,
                                     SparsityEngine &analysis) {
  auto *oper = analysis.getState(op->getOperand(0));
  auto *res = analysis.getState(op->getOpResult(0));

  if (!oper || !res)
    return op.emitError("The lattices are not propagated properly in op: ")
           << op.getOperationName();

  for (std::size_t i = 0; i < res->rank(); i++)
    (*res)[i] = (*oper)[i];

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::DivUnsignedOp op,
                                     SparsityEngine &analysis) {
  auto *oper = analysis.getState(op->getOperand(0));
  auto *res = analysis.getState(op->getOpResult(0));

  if (!oper || !res)
    return op.emitError("The lattices are not propagated properly in op: ")
           << op.getOperationName();

  for (std::size_t i = 0; i < res->rank(); i++)
    (*res)[i] = (*oper)[i];

  return mlir::success();
}
