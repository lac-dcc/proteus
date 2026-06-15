#include "Analysis/SparsityLattice.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

/**
 * @brief Propagates sparsity laterally between operands of the same operation.
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
  static Result run(mlir::Block *block, SparsityAnalysis &analysis);
};

/**
 * @brief Propagates sparsity information backward through a block.
 */
struct BackwardPass {
  /**
   * @brief Runs the backward propagation pass over a block.
   *
   * @param block The MLIR block to analyse.
   * @param analysis The analysis object owning the lattice state to update.
   * @return success() if the pass completed without errors, failure()
   * otherwise.
   */
  static Result run(mlir::Block *block, SparsityAnalysis &analysis);
};

/**
 * @brief Orchestrates the full Sparsity Propagation Analysis (SPA) over a
 * block.
 */
class SparsityAnalysis {
public:
  /**
   * @brief Runs the complete SPA pipeline over a block.
   *
   * Executes Seed → Forward → Lateral → Backward passes in order.
   *
   * @param block The MLIR block to analyse.
   * @return success() if all passes completed without errors, failure()
   *         otherwise.
   */
  mlir::LogicalResult run(mlir::Block *block);

  /**
   * @brief Returns a read-only reference to the full lattice map.
   * @return Const reference to the internal LatticeMap.
   */
  const LatticeMap &getState() const;

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
  const SparsityLattice *getState(const mlir::Value &value) const;

  /**
   * @brief Dispatcher function for inferring sparsity on a specific operation.
   *
   * @param op The operation to visit.
   * @return success() if inference succeeded, failure() otherwise.
   */
  template <typename PassType> Result visit(mlir::Operation &op);

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
  template <typename PassType> Result run(mlir::Block *block);

  LatticeMap state;
};
} // namespace proteus
