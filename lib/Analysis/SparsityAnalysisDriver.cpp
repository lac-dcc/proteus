#include "Analysis/SparsityAnalysisDriver.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"

mlir::LogicalResult proteus::SeedPass::run(mlir::Block *block,
                                           LatticeMap &state) {

  // Check for attributes in the arguments of a function
  if (block->isEntryBlock()) {
    auto *parentOp = block->getParentOp();
    auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(parentOp);
    if (funcOp)
      for (auto &arg : block->getArguments()) {
        auto dict = funcOp.getArgAttrDict(arg.getArgNumber());
        // Reconstructure the lattice from an attribute
        if (dict) {
          auto lattice = SparsityLattice::fromAttr(dict);
          if (lattice) {
            state.try_emplace(arg, lattice.value());
          } else {
            auto lattice = SparsityLattice::defaultFromValue(
                llvm::dyn_cast<mlir::Value>(arg));

            if (lattice.has_value())
              state.try_emplace(arg, lattice.value());
          }
          // Create a default lattice in case of no proteus lattice
          // attribute existing in the block argument
        } else {
          auto lattice = SparsityLattice::defaultFromValue(
              llvm::dyn_cast<mlir::Value>(arg));

          if (lattice.has_value())
            state.try_emplace(arg, lattice.value());
        }
      }
  }

  // TODO: Returns success always currently because we just bypass in the
  // case of no attribute existing in the function arguments
  return mlir::success();
}

mlir::LogicalResult proteus::ForwardPass::run(mlir::Block *block,
                                              LatticeMap &state) {
  for (auto &op : block->getOperations()) {
    // TODO: visit each operation here
  }

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
    // TODO: crash and output something, currently seeding will never fail
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
mlir::LogicalResult proteus::SparsityAnalysis::run(mlir::Block *block) {
  return PassType::run(block, state);
}
