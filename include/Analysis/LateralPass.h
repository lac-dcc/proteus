#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"

namespace proteus {

class SparsityEngine;

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
  static void run(mlir::Block *block, SparsityEngine &analysis);

  /**
   * @brief Dispatcher function for inferring lateral sparsity on a specific
   * operation.
   *
   * @param op The operation to visit.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visit(mlir::Operation &op, SparsityEngine &analysis);

private:
  /**
   * @brief Infers lateral sparsity for a linalg.add operation.
   *
   * @param op The add op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visitOp(mlir::linalg::AddOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a linalg.matmul operation.
   *
   * @param op The matmul op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visitOp(mlir::linalg::MatmulOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a linalg.matvec operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visitOp(mlir::linalg::MatvecOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a linalg.vecmat operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visitOp(mlir::linalg::VecmatOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a linalg.batch_matmul operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visitOp(mlir::linalg::BatchMatmulOp &op,
                      SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a linalg.conv_2d operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visitOp(mlir::linalg::Conv2DOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a linalg.conv_2d_nchw_fchw operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visitOp(mlir::linalg::Conv2DNchwFchwOp &op,
                      SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a linalg.conv_2d_nhwc_hwcf operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visitOp(mlir::linalg::Conv2DNhwcHwcfOp &op,
                      SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for linalg.depthwise_conv_2d_nchw_chw
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visitOp(mlir::linalg::DepthwiseConv2DNchwChwOp &op,
                      SparsityEngine &analysis);

  /**
   * @brief Infers lateral sparsity for a tensor.concat operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return success() if inference succeeded, failure() otherwise.
   */
  static void visitOp(mlir::tensor::ConcatOp &op, SparsityEngine &analysis);
};
} // namespace proteus
