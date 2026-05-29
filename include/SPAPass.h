#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace spa {

struct SPAPass
    : public mlir::PassWrapper<SPAPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const override { return "spa"; }
  llvm::StringRef getDescription() const override {
    return "Tensor Slice Sparsity Propagation Pass";
  }
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createSPAPass();
} // namespace spa
