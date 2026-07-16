#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace proteus {

/**
 * @brief Plants `probe.observe`/`probe.report` calls so every statically
 * predicted result can be checked against its runtime-observed lattice.
 */
struct AddProbeCallsPass : public mlir::OperationPass<mlir::func::FuncOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddProbeCallsPass)

  AddProbeCallsPass()
      : mlir::OperationPass<mlir::func::FuncOp>(
            mlir::TypeID::get<AddProbeCallsPass>()) {}

  [[nodiscard]] llvm::StringRef getName() const override {
    return "AddProbeCallsPass";
  }
  [[nodiscard]] llvm::StringRef getArgument() const override {
    return "add-probe-calls";
  }
  [[nodiscard]] llvm::StringRef getDescription() const override {
    return "Plants probe.observe/probe.report on proteus.lattice-annotated "
           "results";
  }
  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  [[nodiscard]] std::unique_ptr<mlir::Pass> clonePass() const override {
    auto pass = std::make_unique<AddProbeCallsPass>();
    pass->copyOptionValuesFrom(this);
    return pass;
  }
};

std::unique_ptr<mlir::Pass> createAddProbeCallsPass();

} // namespace proteus
