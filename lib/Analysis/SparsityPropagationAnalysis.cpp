#include "Analysis/SparsityPropagationAnalysis.h"

#include "mlir/Pass/Pass.h"

namespace spa {
void SparsityPropagationAnalysis::runOnOperation() {
  getOperation().walk([](mlir::Operation *op) {});
}

std::unique_ptr<mlir::Pass> createSPAPass() {
  return std::make_unique<spa::SparsityPropagationAnalysis>();
}
} // namespace spa
