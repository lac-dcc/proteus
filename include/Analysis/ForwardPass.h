#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"

using Result = mlir::LogicalResult;

namespace proteus {

class SparsityAnalysis;

/**
 * @brief Propagates sparsity information forward through a block.
 */
struct ForwardPass {
  /**
   * @brief Runs the forward propagation pass over a block.
   *
   * @param block The MLIR block to analyse.
   * @param analysis The analysis object owning the lattice state to update.
   * @return success() if the pass completed without errors, failure()
   * otherwise.
   */
  static Result run(mlir::Block *block, SparsityAnalysis &analysis);

  /**
   * @brief Dispatcher function for inferring forward sparsity on a specific
   * operation.
   *
   * @param op The operation to visit.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visit(mlir::Operation &op, SparsityAnalysis &analysis);

private:
  /**
   * @brief Infers forward sparsity for a linalg.add operation.
   *
   * @param op The add op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::AddOp op, SparsityAnalysis &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.matmul operation.
   *
   * @param op The matmul op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::MatmulOp op, SparsityAnalysis &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.abs operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::AbsOp op, SparsityAnalysis &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.ceil operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::CeilOp op, SparsityAnalysis &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.floor operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::FloorOp op, SparsityAnalysis &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.negf operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::NegfOp op, SparsityAnalysis &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.div operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::DivOp op, SparsityAnalysis &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.div_unsigned operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::DivUnsignedOp op,
                        SparsityAnalysis &analysis);
};
} // namespace proteus
