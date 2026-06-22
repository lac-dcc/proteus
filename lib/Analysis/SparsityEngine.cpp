#include "Analysis/SparsityEngine.h"

#include "Analysis/BackwardPass.h"
#include "Analysis/ForwardPass.h"
#include "Analysis/LateralPass.h"
#include "Analysis/SeedPass.h"

#include "mlir/IR/Value.h"

void proteus::SparsityEngine::run(mlir::Block *block) {
  run<SeedPass>(block);
  run<ForwardPass>(block);
  run<LateralPass>(block);
  run<BackwardPass>(block);
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
void proteus::SparsityEngine::run(mlir::Block *block) {
  PassType::run(block, *this);
}

template <typename PassType>
void proteus::SparsityEngine::visit(mlir::Operation &op) {
  PassType::visit(op, *this);
}

template void
proteus::SparsityEngine::visit<proteus::ForwardPass>(mlir::Operation &);
