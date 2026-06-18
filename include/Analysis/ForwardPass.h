#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"

using Result = mlir::LogicalResult;

namespace proteus {

class SparsityEngine;

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
  static Result run(mlir::Block *block, SparsityEngine &analysis);

  /**
   * @brief Dispatcher function for inferring forward sparsity on a specific
   * operation.
   *
   * @param op The operation to visit.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visit(mlir::Operation &op, SparsityEngine &analysis);

private:
  /**
   * @brief Infers forward sparsity for a linalg.add operation.
   *
   * @param op The add op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::AddOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.matmul operation.
   *
   * @param op The matmul op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::MatmulOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.matvec operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::MatvecOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.vecmat operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::VecmatOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.transpose operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::TransposeOp &op,
                        SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.batch_matmul operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::BatchMatmulOp &op,
                        SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.fill operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::FillOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.broadcast operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::BroadcastOp &op,
                        SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a tensor.pad operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::tensor::PadOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a tensor.concat operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::tensor::ConcatOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for ops that pass sparsity through unchanged
   * from their single operand to their result (e.g. abs, ceil, tanh).
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitPassthroughOp(mlir::Operation &op,
                                   SparsityEngine &analysis);
};
} // namespace proteus
