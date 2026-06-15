#include "Analysis/LateralPass.h"

#include "mlir/IR/Value.h"

Result proteus::LateralPass::run(mlir::Block *block, SparsityEngine &analysis) {
  return mlir::success();
}
