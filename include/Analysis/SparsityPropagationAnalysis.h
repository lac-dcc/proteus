#pragma once

#include "Analysis/SparsityEngine.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace proteus {

struct SparsityPropagationAnalysis
    : public mlir::OperationPass<mlir::func::FuncOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparsityPropagationAnalysis)

  SparsityPropagationAnalysis()
      : mlir::OperationPass<mlir::func::FuncOp>(
            mlir::TypeID::get<SparsityPropagationAnalysis>()) {}
  [[nodiscard]] llvm::StringRef getName() const override {
    return "SparsityPropagationAnalysis";
  }
  [[nodiscard]] llvm::StringRef getArgument() const override {
    return "spa-analysis";
  }
  [[nodiscard]] llvm::StringRef getDescription() const override {
    return "Tensor Slice Sparsity Propagation Analysis";
  }
  void runOnOperation() override;

  [[nodiscard]] std::unique_ptr<mlir::Pass> clonePass() const override {
    auto pass = std::make_unique<SparsityPropagationAnalysis>();
    pass->copyOptionValuesFrom(this);
    return pass;
  }

  static std::optional<PassStage> getPassStage(llvm::StringRef name);

  Pass::Option<bool> latticeDump{
      *this, "lattice-dump",
      llvm::cl::desc("Debug utility for showcasing lattices on results."),
      llvm::cl::init(false)};

  Pass::Option<bool> stateDump{
      *this, "state-dump",
      llvm::cl::desc("Debug utility for showcasing DenseMap Lattice state."),
      llvm::cl::init(false)};

  Pass::Option<bool> printZeros{
      *this, "print-zeros",
      llvm::cl::desc("Print a per-value zero-count table after analysis."),
      llvm::cl::init(false)};

  Pass::Option<std::string> passStage{
      *this, "pass-stage",
      llvm::cl::desc("Stop after this pass: seed, forward, lateral, backward."),
      llvm::cl::init("backward")};
};

std::unique_ptr<mlir::Pass> createSparsityPropagationAnalysis();
} // namespace proteus
