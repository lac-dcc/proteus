#include "Transforms/SparsityRewrite.h"

#include "Transforms/Conv2dNchwFchwRewrite.h"
#include "Transforms/MatmulRewrite.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace proteus {

void SparsityRewritePass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  if (target == "linalg") {
    patterns.add<MatmulSparsityLinalgRewritePattern>(patterns.getContext(),
                                                     numMatmulRewrites);
    patterns.add<Conv2dSparsityLinalgRewritePattern>(patterns.getContext(),
                                                     numConv2dRewrites);
  } else {
    patterns.add<MatmulSparsityScfRewritePattern>(patterns.getContext(),
                                                  numMatmulRewrites);
    patterns.add<Conv2dSparsityScfRewritePattern>(patterns.getContext(),
                                                  numConv2dRewrites);
  }

  if (mlir::failed(
          mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }

  if (countRewrites) {
    llvm::errs() << "spa-rewrite counts for @" << getOperation().getSymName()
                 << ": linalg.matmul=" << numMatmulRewrites
                 << ", linalg.conv_2d_nchw_fchw=" << numConv2dRewrites << "\n";
  }
}

std::unique_ptr<mlir::Pass> createSparsityRewritePass() {
  return std::make_unique<SparsityRewritePass>();
}

} // namespace proteus
