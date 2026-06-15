#include "mlir/IR/Value.h"

using Result = mlir::LogicalResult;

namespace proteus {

class SparsityAnalysis;

/**
 * @brief Propagates sparsity information laterally through a block.
 */
struct LateralPass {
  /**
   * @brief Runs the lateral propagation pass over a block.
   *
   * @param block The MLIR block to analyse.
   * @param analysis The analysis object owning the lattice state to update.
   * @return success() if the pass completed without errors, failure()
   * otherwise.
   */
  static Result run(mlir::Block *block, SparsityAnalysis &analysis);
};
} // namespace proteus
