#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
  static Result visitOp(mlir::linalg::AddOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.matmul operation.
   *
   * @param op The matmul op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::MatmulOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.abs operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::AbsOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.ceil operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::CeilOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.floor operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::FloorOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.negf operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::NegFOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.div operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::DivOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.div_unsigned operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::DivUnsignedOp op,
                        SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.copy operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::CopyOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.tanh operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::TanhOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.square operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::SquareOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.square operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::SqrtOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.matvec operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::MatvecOp op, SparsityEngine &analysis);

  /**
   * @brief Infers forward sparsity for a linalg.vecmat operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static Result visitOp(mlir::linalg::VecmatOp op, SparsityEngine &analysis);
};
} // namespace proteus
