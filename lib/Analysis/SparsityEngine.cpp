#include "Analysis/SparsityEngine.h"

#include "Analysis/BackwardPass.h"
#include "Analysis/ForwardPass.h"
#include "Analysis/LateralPass.h"
#include "Analysis/SeedPass.h"

#include "mlir/IR/Value.h"

Result proteus::SparsityEngine::run(mlir::Block *block) {
  if (run<SeedPass>(block).failed()) {
    llvm::errs() << "[SparsityEngine] SeedPass failed on block: "
                 << block->getParentOp()->getName() << "\n";
    llvm_unreachable("SeedPass should never fail");
  }

  if (run<ForwardPass>(block).failed()) {
    llvm::errs() << "[SparsityEngine] ForwardPass failed on block: "
                 << block->getParentOp()->getName() << "\n";
    llvm_unreachable("ForwardPass should never fail");
  }

  if (run<LateralPass>(block).failed()) {
    llvm::errs() << "[SparsityEngine] LateralPass failed on block: "
                 << block->getParentOp()->getName() << "\n";
    llvm_unreachable("LateralPass should never fail");
  }

  if (run<BackwardPass>(block).failed()) {
    llvm::errs() << "[SparsityEngine] BackwardPass failed on block: "
                 << block->getParentOp()->getName() << "\n";
    llvm_unreachable("BackwardPass should never fail");
  }

  return mlir::success();
}

const proteus::LatticeMap &proteus::SparsityEngine::getState() const {
  return state;
}

proteus::LatticeMap &proteus::SparsityEngine::getState() { return state; }

proteus::SparsityLattice *
proteus::SparsityEngine::getState(const mlir::Value &value) {
  auto it = state.find(value);

  if (it != state.end()) {
    return &it->second;
  }

  return nullptr;
}

const proteus::SparsityLattice *
proteus::SparsityEngine::getState(const mlir::Value &value) const {
  auto it = state.find(value);

  if (it != state.end()) {
    return &it->second;
  }

  return nullptr;
}

template <typename PassType>
Result proteus::SparsityEngine::run(mlir::Block *block) {
  return PassType::run(block, *this);
}

template <typename PassType>
Result proteus::SparsityEngine::visit(mlir::Operation &op) {
  return PassType::visit(op, *this);
}

template Result
proteus::SparsityEngine::visit<proteus::ForwardPass>(mlir::Operation &);
