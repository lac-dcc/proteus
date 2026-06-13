#include "Analysis/SparsityPropagationAnalysis.h"

#include "Analysis/SparsityAnalysisDriver.h"
#include "Analysis/SparsityLattice.h"
#include "Analysis/Utilities.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace proteus {
void SparsityPropagationAnalysis::runOnOperation() {

  SparsityAnalysis SA;

  for (auto &block : getOperation().getBody())
    if (SA.run(&block).failed())
      return;

  getOperation().walk([&](mlir::Operation *op) {
    for (auto value : op->getOperands()) {
      auto lattice = SparsityLattice::defaultFromValue(value);

      // Add the value to the block's lattice map
      if (lattice.has_value())
        SA.getState().try_emplace(value, lattice.value());

      if (latticeDump)
        op->setAttr("proteus.lattice",
                    SparsityLattice::toAttr(lattice.value(), op->getContext()));
    }

    if (op->getNumResults() == 1) {
      auto lattice = SparsityLattice::defaultFromValue(op->getResult(0));

      if (latticeDump && lattice.has_value())
        op->setAttr("proteus.lattice",
                    SparsityLattice::toAttr(lattice.value(), op->getContext()));
    }
  });

  if (stateDump)
    printState(SA.getState());
}

std::unique_ptr<mlir::Pass> createSparsityPropagationAnalysis() {
  return std::make_unique<SparsityPropagationAnalysis>();
}
} // namespace proteus
