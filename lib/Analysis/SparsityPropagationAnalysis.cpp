#include "Analysis/SparsityPropagationAnalysis.h"

#include "Analysis/SparsityEngine.h"
#include "Analysis/SparsityLattice.h"
#include "Analysis/Utilities.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace proteus {
void SparsityPropagationAnalysis::runOnOperation() {

  SparsityEngine SA;

  for (auto &block : getOperation().getBody())
    if (SA.run(&block).failed())
      return;

  getOperation().walk([&](mlir::Operation *op) {
    if (op->getNumResults() == 1) {
      auto *lattice = SA.getState(op->getResult(0));

      if (latticeDump && lattice)
        op->setAttr("proteus.lattice",
                    SparsityLattice::toAttr(*lattice, op->getContext()));
    }
  });

  if (stateDump)
    printState(SA.getState());
}

std::unique_ptr<mlir::Pass> createSparsityPropagationAnalysis() {
  return std::make_unique<SparsityPropagationAnalysis>();
}
} // namespace proteus
