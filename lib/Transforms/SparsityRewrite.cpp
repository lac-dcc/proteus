#include "Transforms/SparsityRewrite.h"

#include "Transforms/MatmulRewrite.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace proteus {

void SparsityRewritePass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  if (target == "linalg") {
    patterns.add<MatmulSparsityLinalgRewritePattern>(patterns.getContext());
  } else {
    patterns.add<MatmulSparsityScfRewritePattern>(patterns.getContext());
  }

  if (mlir::failed(
          mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createSparsityRewritePass() {
  return std::make_unique<SparsityRewritePass>();
}

} // namespace proteus
