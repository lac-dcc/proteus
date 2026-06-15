#include "Analysis/SeedPass.h"

#include "Analysis/SparsityAnalysisDriver.h"
#include "Analysis/SparsityLattice.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"

Result proteus::SeedPass::run(mlir::Block *block, SparsityAnalysis &analysis) {
  LatticeMap &state = analysis.getState();

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
