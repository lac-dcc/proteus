#include "Analysis/ForwardPass.h"

#include "Analysis/SparsityEngine.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
                  mlir::linalg::MatvecOp, mlir::linalg::VecmatOp,
                  mlir::linalg::TransposeOp, mlir::linalg::BatchMatmulOp,
                  mlir::linalg::FillOp, mlir::linalg::BroadcastOp>(
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

Result proteus::ForwardPass::visitOp(mlir::linalg::TransposeOp &op,
                                     SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getOperand(0));
  auto *res = analysis.getState(op->getResult(0));

  if ((input == nullptr) || (res == nullptr)) {
    return op.emitError("The lattices are not propagated properly in op: ")
           << mlir::linalg::TransposeOp::getOperationName();
  }

  llvm::ArrayRef<int64_t> perm = op.getPermutation();
  for (std::size_t i = 0; i < res->rank(); i++) {
    (*res)[i] = (*input)[perm[i]];
  }

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::BatchMatmulOp &op,
                                     SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op->getOperand(0));
  auto *rhs = analysis.getState(op->getOperand(1));
  auto *res = analysis.getState(op->getResult(0));

  if ((lhs == nullptr) || (rhs == nullptr) || (res == nullptr)) {
    return op.emitError("The lattices are not propagated properly in op: ")
           << mlir::linalg::BatchMatmulOp::getOperationName();
  }

  // Transfer sparsity for the batch dimension
  (*res)[0] &= (*lhs)[0];
  (*res)[0] &= (*rhs)[0];
  // Row slice sparsity is maintained from the lhs
  (*res)[1] &= (*lhs)[1];
  // Column slice sparsity in maintained from the rhs
  (*res)[2] &= (*rhs)[2];

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::FillOp &op,
                                     SparsityEngine &analysis) {
  auto *res = analysis.getState(op.getResult(0));
  mlir::Value cst = op.getInputs()[0];
  auto cstOp = cst.getDefiningOp<mlir::arith::ConstantOp>();

  if (res == nullptr || !cstOp) {
    return op.emitError("The lattices are not propagated properly in op: ")
           << mlir::linalg::FillOp::getOperationName();
  }

  bool isZero = false;

  // Depending on the type of the constant we have to cast differently
  // In case of FloatType we can get the value of constant through isZero()
  if (mlir::isa<mlir::FloatType>(cst.getType())) {
    isZero = mlir::cast<mlir::FloatAttr>(cstOp.getValue()).getValue().isZero();
    // In case of IntegerType we have to equate
  } else if (mlir::isa<mlir::IntegerType>(cst.getType())) {
    isZero = mlir::cast<mlir::IntegerAttr>(cstOp.getValue()).getInt() == 0;
  }

  // If the constant is zero after all, reset all bits in all ranks
  if (isZero) {
    for (std::size_t i = 0; i < res->rank(); ++i) {
      (*res)[i].reset();
    }
  }

  return mlir::success();
}

Result proteus::ForwardPass::visitOp(mlir::linalg::BroadcastOp &op,
                                     SparsityEngine &analysis) {
  auto *input = analysis.getState(op.getInput());
  auto *res = analysis.getState(op->getResult(0));

  if ((input == nullptr) || (res == nullptr)) {
    return op.emitError("The lattices are not propagated properly in op: ")
           << mlir::linalg::BroadcastOp::getOperationName();
  }

  // Get the dimensions attribute from the op, these dimensions are essentially
  // all the new dimensions that will be added to the tensor
  auto broadcastDims = op.getDimensions();

  // Broadcast does not allow for the splitting or reshaping of the existing
  // tensor that is being broadcasted, so we will keep a counter of the
  // dimension that we last changed and increment it as we iterate through the
  // ranks of the resulting tensor
  std::size_t inputDim = 0;

  // Iterate through the ranks of our resulting tensor
  for (std::size_t i = 0; i < res->rank(); ++i) {
    // In the case of this dimension not being a part of the dimensions being
    // added by the broadcast continue to next rank
    if (llvm::is_contained(broadcastDims, i)) {
      continue;
    }

    // If we do get to a dimension that existed previously in the input tensor
    // we propagate the sparsity to the current rank's bitvector
    (*res)[i] = (*input)[inputDim++];
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
