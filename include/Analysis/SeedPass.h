#include "Analysis/SparsityLattice.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace proteus {

class SparsityEngine;

/**
 * @brief Seeds the lattice state from block arguments and constants.
 */
struct SeedPass {
  /**
   * @brief Runs the seed pass over a block.
   *
   * Seeds block arguments from spa.sparsity annotations.
   *
   * @param block The MLIR block whose arguments are inspected.
   * @param analysis The analysis object owning the lattice state to update.
   * @return success() if seeding completed without errors, failure() otherwise.
   */
  static void run(mlir::Block *block, SparsityEngine &analysis);

  /**
   * @brief Seeds the lattice for a single arith.constant op from its value.
   *
   * For splat constants all slices are sparse when the value is zero. For
   * non-splat constants each slice is sparse iff every element in that slice
   * is zero. Called by the ForwardPass as it visits each op.
   *
   * @param op       The constant op to seed.
   * @param analysis The analysis object owning the lattice state to update.
   */
  static void seedConstant(mlir::arith::ConstantOp op,
                           SparsityEngine &analysis);

  /**
   * @brief Marks the slice index for each dimension that contains the element
   * at linear position @p index as dense.
   *
   * Recovers the multi-dimensional index from the flat element index using
   * row-major @p strides, then sets the corresponding bit in each dimension's
   * BitVector of @p lattice.
   *
   * Shared with proteus::observeMemref, which performs the same row-major
   * decomposition over a runtime memref buffer instead of a DenseElementsAttr.
   *
   * @param index   Flat (row-major) element index.
   * @param strides Row-major strides computed from the tensor shape.
   * @param lattice The lattice to update in-place.
   */
  static void markSlices(uint64_t index, llvm::SmallVector<uint64_t> &strides,
                         SparsityLattice &lattice);

private:
  /**
   * @brief Seeds a splat constant: clears all dimension bits when the splat
   * value is zero.
   *
   * @param denseAttr
   */
  static void seedSplat(mlir::DenseElementsAttr &denseAttr, bool isFloat,
                        SparsityLattice &lattice);
};

} // namespace proteus
