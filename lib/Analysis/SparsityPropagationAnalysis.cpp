#include "Analysis/SparsityPropagationAnalysis.h"

#include "Analysis/SparsityEngine.h"
#include "Analysis/SparsityLattice.h"
#include "Analysis/Utilities.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace proteus {
void SparsityPropagationAnalysis::runOnOperation() {

  SparsityEngine sa;

  for (auto &block : getOperation().getBody()) {
    if (sa.run(&block).failed()) {
      return;
    }
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
}

std::unique_ptr<mlir::Pass> createSparsityPropagationAnalysis() {
  return std::make_unique<SparsityPropagationAnalysis>();
}
} // namespace proteus
