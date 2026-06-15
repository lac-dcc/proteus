#include "Analysis/SparsityAnalysisDriver.h"

#include "Analysis/BackwardPass.h"
#include "Analysis/ForwardPass.h"
#include "Analysis/LateralPass.h"
#include "Analysis/SeedPass.h"

#include "mlir/IR/Value.h"

Result proteus::SparsityAnalysis::run(mlir::Block *block) {
  if (run<SeedPass>(block).failed()) {
    llvm::errs() << "[SparsityAnalysis] SeedPass failed on block: "
                 << block->getParentOp()->getName() << "\n";
    llvm_unreachable("SeedPass should never fail");
  }

  if (run<ForwardPass>(block).failed()) {
    llvm::errs() << "[SparsityAnalysis] ForwardPass failed on block: "
                 << block->getParentOp()->getName() << "\n";
    llvm_unreachable("ForwardPass should never fail");
  }

  if (run<LateralPass>(block).failed()) {
    llvm::errs() << "[SparsityAnalysis] LateralPass failed on block: "
                 << block->getParentOp()->getName() << "\n";
    llvm_unreachable("LateralPass should never fail");
  }

  if (run<BackwardPass>(block).failed()) {
    llvm::errs() << "[SparsityAnalysis] BackwardPass failed on block: "
                 << block->getParentOp()->getName() << "\n";
    llvm_unreachable("BackwardPass should never fail");
  }

  return mlir::success();
}

const proteus::LatticeMap &proteus::SparsityAnalysis::getState() const {
  return state;
}

proteus::LatticeMap &proteus::SparsityAnalysis::getState() { return state; }

proteus::SparsityLattice *
proteus::SparsityAnalysis::getState(const mlir::Value &value) {
  auto it = state.find(value);

  if (it != state.end())
    return &it->second;

  return nullptr;
}

const proteus::SparsityLattice *
proteus::SparsityAnalysis::getState(const mlir::Value &value) const {
  auto it = state.find(value);

  if (it != state.end())
    return &it->second;

  return nullptr;
}

template <typename PassType>
Result proteus::SparsityAnalysis::run(mlir::Block *block) {
  return PassType::run(block, *this);
}

template <typename PassType>
Result proteus::SparsityAnalysis::visit(mlir::Operation &op) {
  return PassType::visit(op, *this);
}

template Result
proteus::SparsityAnalysis::visit<proteus::ForwardPass>(mlir::Operation &);
