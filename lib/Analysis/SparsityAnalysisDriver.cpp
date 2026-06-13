#include "Analysis/SparsityAnalysisDriver.h"

#include "mlir/IR/Value.h"

mlir::LogicalResult proteus::SeedPass::run(mlir::Block *block,
                                           LatticeMap &state) {
  return mlir::success();
}

mlir::LogicalResult proteus::ForwardPass::run(mlir::Block *block,
                                              LatticeMap &state) {
  return mlir::success();
}

mlir::LogicalResult proteus::LateralPass::run(mlir::Block *block,
                                              LatticeMap &state) {
  return mlir::success();
}

mlir::LogicalResult proteus::BackwardPass::run(mlir::Block *block,
                                               LatticeMap &state) {
  return mlir::success();
}

mlir::LogicalResult proteus::SparsityAnalysis::run(mlir::Block *block) {
  if (run<SeedPass>(block).failed()) {
    // TODO: crash and output something
  }

  if (run<ForwardPass>(block).failed()) {
    // TODO: crash and output something
  }

  if (run<LateralPass>(block).failed()) {
    // TODO: crash and output something
  }

  if (run<BackwardPass>(block).failed()) {
    // TODO: crash and output something
  }

  return mlir::success();
}

proteus::LatticeMap &proteus::SparsityAnalysis::getState() { return state; }

const proteus::LatticeMap &proteus::SparsityAnalysis::getState() const {
  return state;
}

template <typename PassType>
mlir::LogicalResult proteus::SparsityAnalysis::run(mlir::Block *block) {
  return PassType::run(block, state);
}
