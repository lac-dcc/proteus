#include "Analysis/BackwardPass.h"

#include "mlir/IR/Value.h"

Result proteus::BackwardPass::run(mlir::Block *block,
                                  SparsityEngine &analysis) {
  return mlir::success();
}
