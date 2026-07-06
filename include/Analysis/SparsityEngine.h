#pragma once

#include "Analysis/SparsityLattice.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/Timing.h"

using Result = mlir::LogicalResult;

namespace mlir {
class TimingScope;
}

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
   * @param timing If non-null, each stage is wrapped in a nested
   *        `mlir::TimingScope` under this scope, so per-stage timing shows up
   *        when the caller's `TimingManager` is enabled.
   * @return success() if all passes completed without errors, failure()
   *         otherwise.
   */
  void run(mlir::Block *block, PassStage stage = PassStage::Backward,
           mlir::TimingScope *timing = nullptr);

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

  /*
   * @brief Setter for the candidate value of the internal state of the analysis
   * for the fixpoint algorithm
   *
   * @param value The value to set in the internal state
   */
  void setCandidateValue(const mlir::Value &value);

  /*
   * @brief Getter for the candidate value of the internal state of the analysis
   *
   * @return mlir::Value& The internal state's candidate value
   */
  mlir::Value &getCandidateValue();

  /*
   * @brief Handles nesting when making timing report with time-passes flag in
   * proteus-opt
   *
   * @param name Name of the current scope we are timing
   * @param *timing Pointer to the timing manager
   * @return Returns new timing manager after nesting
   */
  static mlir::TimingScope nestTimeScope(llvm::StringRef name,
                                         mlir::TimingScope *timing);

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
  mlir::Value candidateVal;
};
} // namespace proteus
