#include "mlir/IR/Value.h"

using Result = mlir::LogicalResult;

namespace proteus {

class SparsityAnalysis;

/**
 * @brief Seeds the lattice state from block arguments.
 */
struct SeedPass {
  /**
   * @brief Runs the seed pass over a block.
   *
   * @param block The MLIR block whose arguments are inspected.
   * @param analysis The analysis object owning the lattice state to update.
   * @return success() if seeding completed without errors, failure() otherwise.
   */
  static Result run(mlir::Block *block, SparsityAnalysis &analysis);
};
} // namespace proteus
