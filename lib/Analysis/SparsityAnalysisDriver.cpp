#include "Analysis/SparsityAnalysisDriver.h"

#include "Analysis/ForwardPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

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

Result proteus::LateralPass::run(mlir::Block *block,
                                 SparsityAnalysis &analysis) {
  return mlir::success();
}

Result proteus::BackwardPass::run(mlir::Block *block,
                                  SparsityAnalysis &analysis) {
  return mlir::success();
}

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
