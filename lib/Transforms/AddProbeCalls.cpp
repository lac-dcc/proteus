#include "Transforms/AddProbeCalls.h"

#include "Dialect/Probe/IR/Probe.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace proteus {

void AddProbeCallsPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<mlir::probe::ProbeDialect>();
}

void AddProbeCallsPass::runOnOperation() {
  mlir::func::FuncOp funcOp = getOperation();

  llvm::SmallVector<mlir::Operation *> worklist;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->hasAttr("proteus.lattice")) {
      worklist.push_back(op);
    }
  });

  mlir::OpBuilder builder(&getContext());
  for (auto [opID, op] : llvm::enumerate(worklist)) {
    for (auto [resultID, result] : llvm::enumerate(op->getResults())) {
      builder.setInsertionPointAfter(op);
      builder.create<mlir::probe::ObserveOp>(
          op->getLoc(), result, static_cast<uint32_t>(opID),
          static_cast<uint32_t>(resultID)); // NOLINT
    }
  }

  if (worklist.empty()) {
    return;
  }

  llvm::SmallVector<mlir::func::ReturnOp> returns;
  funcOp.walk(
      [&](mlir::func::ReturnOp returnOp) { returns.push_back(returnOp); });
  for (auto returnOp : returns) {
    builder.setInsertionPoint(returnOp);
    builder.create<mlir::probe::ReportOp>(returnOp->getLoc()); // NOLINT
  }
}

std::unique_ptr<mlir::Pass> createAddProbeCallsPass() {
  return std::make_unique<AddProbeCallsPass>();
}

} // namespace proteus
