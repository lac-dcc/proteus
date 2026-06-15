#include "mlir/IR/Value.h"

using Result = mlir::LogicalResult;

namespace proteus {

class SparsityEngine;

/**
 * @brief Propagates sparsity information backward through a block.
 */
struct BackwardPass {
  /**
   * @brief Runs the backward propagation pass over a block.
   *
   * @param block The MLIR block to analyse.
   * @param analysis The analysis object owning the lattice state to update.
   * @return success() if the pass completed without errors, failure()
   * otherwise.
   */
  static Result run(mlir::Block *block, SparsityEngine &analysis);
};
} // namespace proteus
