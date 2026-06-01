#include "Analysis/SparsityPropagationAnalysis.h"

#include "mlir/Pass/Pass.h"
#include <iostream>

namespace proteus {
void SparsityPropagationAnalysis::runOnOperation() {
  getOperation().walk([](mlir::Operation *op) {});

  if (latticeDump) {
    std::cout << "latticeDump enabled.\n";
  } else {
    std::cout << "latticeDump disabled.\n";
  }
}

std::unique_ptr<mlir::Pass> createSparsityPropagationAnalysis() {
  return std::make_unique<SparsityPropagationAnalysis>();
}
} // namespace proteus
