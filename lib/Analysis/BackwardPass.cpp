#include "Analysis/BackwardPass.h"

#include "mlir/IR/Value.h"

Result proteus::BackwardPass::run(mlir::Block *block,
                                  SparsityAnalysis &analysis) {
  return mlir::success();
}
