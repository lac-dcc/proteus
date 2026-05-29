#include "SPAPass.h"

#include "mlir/Pass/Pass.h"

namespace spa {
void SPAPass::runOnOperation() {
  getOperation().walk([](mlir::Operation *op) {});
}

std::unique_ptr<mlir::Pass> createSPAPass() {
  return std::make_unique<spa::SPAPass>();
}
} // namespace spa
