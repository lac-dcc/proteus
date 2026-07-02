#include "Analysis/LateralPass.h"

#include "Analysis/SeedPass.h"
#include "Analysis/SparsityEngine.h"
#include "Analysis/SparsityLattice.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

void proteus::LateralPass::run(mlir::Block *block, SparsityEngine &analysis) {
  auto worklist = getWorklist(block);

  while (!worklist.empty()) {
    analysis.setCandidateValue(worklist.pop_back_val());

    auto *current = analysis.getState(analysis.getCandidateValue());

    std::optional<SparsityLattice> joined;
    for (mlir::Operation *user : analysis.getCandidateValue().getUsers()) {
      auto candidate = visit(*user, analysis);
      if (!candidate) {
        continue;
      }
      joined = joined ? SparsityLattice::join(*joined, *candidate) : candidate;
    }

    if (!joined) {
      continue;
    }

    auto merged = SparsityLattice::meet(*current, *joined);
    if (merged != *current) {
      *current = merged;

      for (mlir::Operation *user : analysis.getCandidateValue().getUsers()) {
        for (mlir::Value operand : user->getOperands()) {
          if (operand != analysis.getCandidateValue()) {
            worklist.push_back(operand);
          }
        }
      }
    }
  }
}

std::optional<proteus::SparsityLattice>
proteus::LateralPass::visit(mlir::Operation &op, SparsityEngine &analysis) {
  // Based on the mlir operation we dispatch the appropriate transfer function
  return mlir::TypeSwitch<mlir::Operation *, std::optional<SparsityLattice>>(
             &op)
      .Case<mlir::linalg::MatmulOp, mlir::linalg::BatchMatmulOp,
            mlir::linalg::MatvecOp>(
          [&](auto typedOp) { return visitOp(typedOp, analysis); })
      .Default(
          [&](mlir::Operation *defaultOp) -> std::optional<SparsityLattice> {
            if (defaultOp->getNumResults() == 0) {
              return std::nullopt;
            }

            return *analysis.getState(analysis.getCandidateValue());
          });
}

std::optional<proteus::SparsityLattice>
proteus::LateralPass::visitOp(mlir::linalg::MatmulOp &op,
                              SparsityEngine &analysis) {
  if (op.getOperand(0) == op.getOperand(1)) {
    return std::nullopt;
  }

  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));

  if (analysis.getCandidateValue() == op.getOperand(0)) {
    SparsityLattice candidate = *lhs;
    candidate[1] &= (*rhs)[0];
    return candidate;
  }

  if (analysis.getCandidateValue() == op.getOperand(1)) {
    SparsityLattice candidate = *rhs;
    candidate[0] &= (*lhs)[1];
    return candidate;
  }

  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::LateralPass::visitOp(mlir::linalg::BatchMatmulOp &op,
                              SparsityEngine &analysis) {
  if (op.getOperand(0) == op.getOperand(1)) {
    return std::nullopt;
  }

  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));

  if (analysis.getCandidateValue() == op.getOperand(0)) {
    SparsityLattice candidate = *lhs;
    candidate[2] &= (*rhs)[1];
    return candidate;
  }

  if (analysis.getCandidateValue() == op.getOperand(1)) {
    SparsityLattice candidate = *rhs;
    candidate[1] &= (*lhs)[2];
    return candidate;
  }

  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::LateralPass::visitOp(mlir::linalg::MatvecOp &op,
                              SparsityEngine &analysis) {

  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));

  if (analysis.getCandidateValue() == op.getOperand(0)) {
    SparsityLattice candidate = *lhs;
    candidate[1] &= (*rhs)[0];
    return candidate;
  }

  if (analysis.getCandidateValue() == op.getOperand(1)) {
    SparsityLattice candidate = *rhs;
    candidate[0] &= (*lhs)[1];
    return candidate;
  }

  return std::nullopt;
}

llvm::SmallVector<mlir::Value>
proteus::LateralPass::getWorklist(mlir::Block *block) {
  llvm::SmallVector<mlir::Value> worklist;

  for (auto &op : block->getOperations()) {
    if (mlir::isa<mlir::linalg::MatmulOp>(op) ||
        mlir::isa<mlir::linalg::BatchMatmulOp>(op) ||
        mlir::isa<mlir::linalg::MatvecOp>(op)) {
      for (auto operand : op.getOperands()) {
        worklist.push_back(operand);
      }
    }
  }

  return worklist;
}
