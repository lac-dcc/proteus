#pragma once

#include "Analysis/SparsityLattice.h"

#include "mlir/IR/Value.h"

using Result = mlir::LogicalResult;

namespace proteus {

enum class PassStage : uint8_t { Seed, Forward, Lateral, Backward };

class SparsityEngine;

/**
 * @brief Orchestrates the full Sparsity Propagation Analysis (SPA) over a
 * block.
 */
class SparsityEngine {
public:
  /**
   * @brief Runs the complete SPA pipeline over a block.
   *
   * Executes Seed → Forward → Lateral → Backward passes in order.
   *
   * @param block The MLIR block to analyse.
   * @param stage Runs the analysis up to the requested pass
   * @return success() if all passes completed without errors, failure()
   *         otherwise.
   */
  void run(mlir::Block *block, PassStage stage = PassStage::Backward);

  /**
   * @brief Returns a read-only reference to the full lattice map.
   * @return Const reference to the internal LatticeMap.
   */
  [[nodiscard]] const LatticeMap &getState() const;

  /**
   * @brief Returns a mutable reference to the full lattice map.
   * @return Reference to the internal LatticeMap.
   */
  LatticeMap &getState();

  /**
   * @brief Looks up the lattice for a specific MLIR value.
   *
   * @param value The value to query.
   * @return Pointer to the SparsityLattice for @p value, or nullptr if the
   *         value is not present in the map.
   */
  SparsityLattice *getState(const mlir::Value &value);

  /**
   * @brief Looks up the lattice for a specific MLIR value (const overload).
   *
   * @param value The value to query.
   * @return Const pointer to the SparsityLattice for @p value, or nullptr if
   *         the value is not present in the map.
   */
  [[nodiscard]] const SparsityLattice *getState(const mlir::Value &value) const;

  /**
   * @brief Dispatcher function for inferring sparsity on a specific operation.
   *
   * @param op The operation to visit.
   * @return success() if inference succeeded, failure() otherwise.
   */
  template <typename PassType> void visit(mlir::Operation &op);

  /**
   * @brief Infers sparsity for a linalg.matmul operation.
   *
   * @param op The matmul op to analyse.
   * @return success() if inference succeeded, failure() otherwise.
   */
private:
  /**
   * @brief Dispatches a single analysis pass of type @p PassType.
   *
   * @tparam PassType One of SeedPass, ForwardPass, LateralPass, BackwardPass.
   * @param block The block to run the pass over.
   * @return success() or failure() as returned by PassType::run.
   */
  template <typename PassType> void run(mlir::Block *block);

  LatticeMap state;
};
} // namespace proteus
