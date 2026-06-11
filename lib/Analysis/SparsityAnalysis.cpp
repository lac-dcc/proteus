#include "Analysis/SparsityAnalysis.h"

#include "mlir/IR/Value.h"

mlir::LogicalResult proteus::ForwardPass::run(mlir::Block *block,
                                              LatticeMap &state) {
  return mlir::success();
}

mlir::LogicalResult proteus::SparsityAnalysis::run(mlir::Block *block) {
  auto forward_result = run<ForwardPass>(block);
  // auto lateral_result = run<LateralPass>(block);
  // auto backward_result = run<BackwardPass>(block);
  return forward_result;
}

proteus::LatticeMap &proteus::SparsityAnalysis::getState() { return state; }

const proteus::LatticeMap &proteus::SparsityAnalysis::getState() const {
  return state;
}

template <typename PassType>
mlir::LogicalResult proteus::SparsityAnalysis::run(mlir::Block *block) {
  return PassType::run(block, state);
}
