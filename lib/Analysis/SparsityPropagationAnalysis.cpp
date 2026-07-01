#include "Analysis/SparsityPropagationAnalysis.h"

#include "Analysis/SparsityLattice.h"
#include "Analysis/Utilities.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace proteus {

std::optional<PassStage>
SparsityPropagationAnalysis::getPassStage(llvm::StringRef name) {
  if (name == "seed") {
    return PassStage::Seed;
  }
  if (name == "forward") {
    return PassStage::Forward;
  }
  if (name == "lateral") {
    return PassStage::Lateral;
  }
  if (name == "backward") {
    return PassStage::Backward;
  }

  return std::nullopt;
}

void SparsityPropagationAnalysis::runOnOperation() {

  auto stage = getPassStage(passStage);

  if (!stage) {
    getOperation().emitError()
        << "spa-analysis: invalid last-pass value '" << passStage
        << "'; expected one of: seed, forward, lateral, backward";
    signalPassFailure();
    return;
  }

  SparsityEngine sa;
  for (auto &block : getOperation().getBody()) {
    sa.run(&block, *stage);
  }

  getOperation().walk([&](mlir::Operation *op) {
    if (op->getNumResults() == 1) {
      auto *lattice = sa.getState(op->getResult(0));

      if (latticeDump && lattice) {
        op->setAttr("proteus.lattice",
                    SparsityLattice::toAttr(*lattice, op->getContext()));
      }
    }
  });

  if (stateDump) {
    printState(sa.getState());
  }

  if (printZeros) {
    proteus::ZeroCounter::print(sa.getState());
  }
}

std::unique_ptr<mlir::Pass> createSparsityPropagationAnalysis() {
  return std::make_unique<SparsityPropagationAnalysis>();
}
} // namespace proteus
