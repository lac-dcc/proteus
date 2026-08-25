#pragma once

#include "Analysis/SparsityLattice.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

using ZeroMap = llvm::DenseMap<mlir::Value, std::pair<uint64_t, uint64_t>>;

namespace proteus {
void printState(const llvm::DenseMap<mlir::Value, SparsityLattice> &state);

/**
 * @brief Computes the per-dimension dense/sparse bitmap for a runtime memref.
 *
 * A dimension's slice is dense if any element with that index was observed
 * nonzero, sparse if every such element is zero.
 *
 * @param mref The observed tensor's runtime buffer.
 * @return The runtime SparsityLattice describing the buffer's sparsity.
 */
SparsityLattice observeMemref(DynamicMemRefType<float> mref);

/* Keeps track of all the zeroes in the global analysis state */
struct ZeroCounter {
  /**
   * @brief Counts the number of zeroes in a SparsityLattice
   *
   * @param lattice The lattice to analyze.
   * @return Total number of cleared bits across all dimensions and the total
   * number of bits across all dimensions.
   */
  static std::pair<uint64_t, uint64_t> count(const SparsityLattice &lattice);

  /**
   * @brief Counts the number of zeroes in a SparsityMap
   *
   * @param state The analysis state
   * @return A map from each MLIR value to its zero count and total count.
   */
  static ZeroMap count(const LatticeMap &state);

  /**
   * @brief Pretty printer for the analysis state total number of zeroes
   *
   * @param state The full analysis state
   * @param os The output stream to write to.
   */
  static void print(const LatticeMap &state,
                    llvm::raw_ostream &os = llvm::errs());
};
} // namespace proteus
