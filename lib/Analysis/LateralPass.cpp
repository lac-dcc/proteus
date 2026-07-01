#include "Analysis/LateralPass.h"

#include "Analysis/SeedPass.h"
#include "Analysis/SparsityEngine.h"
#include "Analysis/SparsityLattice.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

void proteus::LateralPass::run(mlir::Block *block, SparsityEngine &analysis) {
  auto worklist = SeedPass::lateralWorklist(block);

  while (!worklist.empty()) {
    analysis.candidateVal = worklist.pop_back_val();
    auto *current = analysis.getState(analysis.candidateVal);

    std::optional<SparsityLattice> joined;
    for (mlir::Operation *user : analysis.candidateVal.getUsers()) {
      auto candidate = visit(*user, analysis);
      joined = joined ? SparsityLattice::join(*joined, *candidate) : candidate;
    }

    if (!joined) {
      continue;
    }

    auto merged = SparsityLattice::meet(*current, *joined);
    if (merged != *current) {
      *current = merged;

      for (mlir::Operation *user : analysis.candidateVal.getUsers()) {
        for (mlir::Value operand : user->getOperands()) {
          if (operand != analysis.candidateVal) {
            worklist.insert(operand);
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
      .Case<mlir::linalg::MatmulOp, mlir::linalg::BatchMatmulOp>(
          [&](auto typedOp) { return visitOp(typedOp, analysis); })
      .Default([](auto) { return std::nullopt; });
}

std::optional<proteus::SparsityLattice>
proteus::LateralPass::visitOp(mlir::linalg::MatmulOp &op,
                              SparsityEngine &analysis) {
  auto *lhs = analysis.getState(op.getOperand(0));
  auto *rhs = analysis.getState(op.getOperand(1));

  if (analysis.candidateVal == op.getOperand(0)) {
    SparsityLattice candidate = *lhs;
    candidate[1] &= (*rhs)[0];
    return candidate;
  }

  if (analysis.candidateVal == op.getOperand(1)) {
    SparsityLattice candidate = *rhs;
    candidate[0] &= (*lhs)[1];
    return candidate;
  }
  return std::nullopt;
}

std::optional<proteus::SparsityLattice>
proteus::LateralPass::visitOp(mlir::linalg::BatchMatmulOp &op,
                              SparsityEngine &analysis) {
  auto *rhs = analysis.getState(op.getOperand(1));
  auto *lhs = analysis.getState(op.getOperand(0));

  if (analysis.candidateVal == op.getOperand(0)) {
    SparsityLattice candidate = *lhs;
    candidate[2] &= (*rhs)[1];
    return candidate;
  }
  if (analysis.candidateVal == op.getOperand(1)) {
    SparsityLattice candidate = *rhs;
    candidate[1] &= (*lhs)[2];
    return candidate;
  }
  return std::nullopt;
}
