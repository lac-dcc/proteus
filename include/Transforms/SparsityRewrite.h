#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class RewritePatternSet;
}

namespace proteus {

/**
 * @brief Rewrites ops annotated with `proteus.lattice` to exploit statically
 * proven sparsity.
 */
struct SparsityRewritePass : public mlir::OperationPass<mlir::func::FuncOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparsityRewritePass)

  SparsityRewritePass()
      : mlir::OperationPass<mlir::func::FuncOp>(
            mlir::TypeID::get<SparsityRewritePass>()) {}

  [[nodiscard]] llvm::StringRef getName() const override {
    return "SparsityRewritePass";
  }
  [[nodiscard]] llvm::StringRef getArgument() const override {
    return "spa-rewrite";
  }
  [[nodiscard]] llvm::StringRef getDescription() const override {
    return "Rewrites proteus.lattice annotated ops to exploit proven sparsity";
  }
  void runOnOperation() override;

  [[nodiscard]] std::unique_ptr<mlir::Pass> clonePass() const override {
    auto pass = std::make_unique<SparsityRewritePass>();
    pass->copyOptionValuesFrom(this);
    return pass;
  }

  Pass::Option<std::string> target{
      *this, "target",
      llvm::cl::desc("Rewrite target between linalg (default) or scf."),
      llvm::cl::init("linalg")};

  Pass::Option<bool> countRewrites{
      *this, "count-rewrites",
      llvm::cl::desc(
          "Print, per rewrite rule, how many ops it rewrote in this func."),
      llvm::cl::init(false)};

  unsigned numMatmulRewrites = 0;
  unsigned numConv2dRewrites = 0;
};

std::unique_ptr<mlir::Pass> createSparsityRewritePass();

} // namespace proteus
