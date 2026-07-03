#pragma once

#include "Analysis/SparsityLattice.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"

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
   */
  static void run(mlir::Block *block, SparsityEngine &analysis);

  /**
   * @brief Dispatcher function for inferring backward sparsity on a specific
   * operation.
   *
   * @param op The operation to visit.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice `op`'s transfer function proposes for
   * `value`, or std::nullopt if `op` has no lateral transfer function for
   * that operand.
   */
  static std::optional<proteus::SparsityLattice>
  visit(mlir::Operation &op, SparsityEngine &analysis);

  /**
   * @brief Seeds the backward worklist for the worklist fixedpoint algorithm
   *
   * @param block The MLIR block to analyse.
   * @return The lateral worklist in question
   */
  static llvm::SmallVector<mlir::Value> getWorklist(mlir::Block *block);

private:
  /**
   * @brief Infers backward sparsity for a linalg.matmul operation.
   *
   * @param op The matmul op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::MatmulOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a linalg.add operation.
   *
   * @param op The add op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::AddOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a linalg.matvec operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::MatvecOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a linalg.vecmat operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::VecmatOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a linalg.transpose operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::TransposeOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a linalg.batch_matmul operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::BatchMatmulOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a linalg.fill operation.
   *
   * The fill value is a scalar constant; there is no tensor operand to
   * propagate backward to.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::FillOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a linalg.broadcast operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::BroadcastOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a tensor.pad operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::tensor::PadOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a tensor.concat operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::tensor::ConcatOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a tensor.empty operation.
   *
   * There is nothing to implement here, this is added to just indicate the
   * support for the operation. The default behaviour will suffice.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::tensor::EmptyOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a linalg.conv_2d operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::Conv2DOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a linalg.conv_2d_nchw_fchw operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::Conv2DNchwFchwOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a linalg.conv_2d_nhwc_hwcf operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::Conv2DNhwcHwcfOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for linalg.depthwise_conv_2d_nchw_chw.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::DepthwiseConv2DNchwChwOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for linalg.pooling_nchw_max.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::PoolingNchwMaxOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for linalg.pooling_nchw_sum.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::linalg::PoolingNchwSumOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a tensor.expand_shape operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::tensor::ExpandShapeOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a tensor.extract_slice operation.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::tensor::ExtractSliceOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for a tensor.collapse_shape operation.
   *
   * A sparse result only implies that at least one contributing input fiber
   * is sparse, but not which one; no backward propagation is possible.
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static std::optional<proteus::SparsityLattice>
  visitOp(mlir::tensor::CollapseShapeOp &op, SparsityEngine &analysis);

  /**
   * @brief Infers backward sparsity for ops that pass sparsity through
   * unchanged from their result to their single operand (e.g. abs, ceil,
   * tanh).
   *
   * @param op The op to analyse.
   * @param analysis The SPA analysis object.
   * @return The candidate lattice for `value`, or std::nullopt if `value` is
   * not the lhs or rhs operand.
   */
  static SparsityLattice visitPassthroughOp(mlir::Operation &op,
                                            SparsityEngine &analysis);
};

} // namespace proteus
