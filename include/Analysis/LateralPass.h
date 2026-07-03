#include "Analysis/SparsityLattice.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace proteus {

class SparsityEngine;

/**
 * @brief Propagates sparsity information laterally through a block.
 */
struct LateralPass {
  /**
   * @brief Runs the lateral propagation pass over a block.
   *
   * Uses a worklist fixpoint algorithm, whenever a value's lattice is
   * tightened, every other operand of the ops that read it is enqueued, until
   * no more changes occur.
   *
   * @param block The MLIR block to analyse.
   * @param analysis The analysis object owning the lattice state to update.
   */
  static void run(mlir::Block *block, SparsityEngine &analysis);

  /**
   * @brief Dispatcher that proposes a candidate lattice for `value`
   *
   * @param op An operation that reads `value`.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice `op`'s transfer function proposes for
   * `value`, or std::nullopt if `op` has no lateral transfer function for
   * that operand.
   */
  static std::optional<SparsityLattice> visit(mlir::Operation &op,
                                              SparsityEngine &analysis);

  /**
   * @brief Seeds the lateral worklist for the worklist fixedpoint algorithm
   *
   * @param block The MLIR block to analyse.
   * @return The lateral worklist in question
   */
  static llvm::SmallVector<mlir::Value> getWorklist(mlir::Block *block);

private:
  /**
   * @brief Infers lateral sparsity for a linalg.matmul operation.
   *
   * Proposes a new candidate lattice for whichever of lhs/rhs `value` is.
   *
   * @param op The matmul op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<SparsityLattice> visitOp(mlir::linalg::MatmulOp &op,
                                                SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a linalg.batch_matmul operation.
   *
   * Proposes a new candidate lattice for whichever of lhs/rhs `value` is.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<SparsityLattice> visitOp(mlir::linalg::BatchMatmulOp &op,
                                                SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a linalg.matvec operation.
   *
   * Proposes a new candidate lattice for whichever of lhs/rhs `value` is.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<SparsityLattice> visitOp(mlir::linalg::MatvecOp &op,
                                                SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a linalg.vecmat operation.
   *
   * Proposes a new candidate lattice for whichever of lhs/rhs `value` is.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<SparsityLattice> visitOp(mlir::linalg::VecmatOp &op,
                                                SparsityEngine &analysis);
};
} // namespace proteus
