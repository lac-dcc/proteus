#include "Analysis/SparsityPropagationAnalysis.h"

#include "Analysis/SparsityLattice.h"

#include "mlir/Pass/Pass.h"

namespace proteus {
void SparsityPropagationAnalysis::runOnOperation() {

  getOperation().walk([](mlir::Operation *op) {
    auto lattice = SparsityLattice::getSparsityLattice(op);

    if (lattice) {
      op->emitWarning(
          "This is an op that should have a lattice or something right?");
    }
  });
}

std::unique_ptr<mlir::Pass> createSparsityPropagationAnalysis() {
  return std::make_unique<SparsityPropagationAnalysis>();
}
} // namespace proteus
