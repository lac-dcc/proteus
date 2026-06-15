#include "Analysis/LateralPass.h"

#include "mlir/IR/Value.h"

Result proteus::LateralPass::run(mlir::Block *block,
                                 SparsityAnalysis &analysis) {
  return mlir::success();
}
