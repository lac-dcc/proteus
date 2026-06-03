#include "Analysis/SparsityPropagationAnalysis.h"

#include "Analysis/SparsityLattice.h"

#include "mlir/Pass/Pass.h"

namespace proteus {
void SparsityPropagationAnalysis::runOnOperation() {
  getOperation().walk([this](mlir::Operation *op) {
    auto *lattice = SparsityLattice::fromOp(op);

    if (!lattice)
      return;

    if (latticeDump)
      op->setAttr("spa", SparsityLattice::toAttr(*lattice, op->getContext()));

    delete lattice;
  });
}

std::unique_ptr<mlir::Pass> createSparsityPropagationAnalysis() {
  return std::make_unique<SparsityPropagationAnalysis>();
}
} // namespace proteus
