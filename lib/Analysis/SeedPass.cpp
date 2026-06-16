#include "Analysis/SeedPass.h"

#include "Analysis/SparsityEngine.h"
#include "Analysis/SparsityLattice.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"

static std::optional<proteus::SparsityLattice>
resolveArgLattice(mlir::BlockArgument arg, mlir::func::FuncOp funcOp) {
  auto dict = funcOp.getArgAttrDict(arg.getArgNumber());
  if (dict) {
    if (auto lattice = proteus::SparsityLattice::fromAttr(dict)) {
      return lattice;
    }
  }
  return proteus::SparsityLattice::defaultFromValue(arg);
}

Result proteus::SeedPass::run(mlir::Block *block, SparsityEngine &analysis) {
  if (!block->isEntryBlock()) {
    return mlir::success();
  }

  auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(block->getParentOp());
  if (!funcOp) {
    return mlir::success();
  }

  LatticeMap &state = analysis.getState();
  for (auto &arg : block->getArguments()) {
    auto lattice = resolveArgLattice(arg, funcOp);
    if (lattice.has_value()) {
      state.try_emplace(arg, lattice.value());
    }
  }

  return mlir::success();
}
