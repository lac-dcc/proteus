#include "Analysis/SparsityPropagationAnalysis.h"

#include "mlir/Pass/Pass.h"

namespace proteus {
void SparsityPropagationAnalysis::runOnOperation() {
  getOperation().walk([](mlir::Operation *op) {});
}

std::unique_ptr<mlir::Pass> createSPAPass() {
  return std::make_unique<proteus::SparsityPropagationAnalysis>();
}
} // namespace proteus
