#include "Analysis/SparsityEngine.h"

#include "Analysis/BackwardPass.h"
#include "Analysis/ForwardPass.h"
#include "Analysis/LateralPass.h"
#include "Analysis/SeedPass.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/Timing.h"

void proteus::SparsityEngine::run(mlir::Block *block, PassStage stage,
                                  mlir::TimingScope *timing) {

  auto scope = nestTimeScope("Seed", timing);
  run<SeedPass>(block);

  if (stage == PassStage::Seed) {
    return;
  }

  scope = nestTimeScope("Forward", timing);
  run<ForwardPass>(block);

  if (stage == PassStage::Forward) {
    return;
  }

  scope = nestTimeScope("Lateral", timing);
  run<LateralPass>(block);

  if (stage == PassStage::Lateral) {
    return;
  }

  scope = nestTimeScope("Backward", timing);
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

void proteus::SparsityEngine::setCandidateValue(const mlir::Value &value) {
  candidateVal = value;
}

mlir::Value &proteus::SparsityEngine::getCandidateValue() {
  return candidateVal;
}

mlir::TimingScope
proteus::SparsityEngine::nestTimeScope(llvm::StringRef name,
                                       mlir::TimingScope *timing) {
  return (timing != nullptr) ? timing->nest(name) : mlir::TimingScope();
}

template void
proteus::SparsityEngine::visit<proteus::ForwardPass>(mlir::Operation &);
template void
proteus::SparsityEngine::visit<proteus::LateralPass>(mlir::Operation &);
template void
proteus::SparsityEngine::visit<proteus::BackwardPass>(mlir::Operation &);
