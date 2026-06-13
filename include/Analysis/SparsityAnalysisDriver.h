#include "Analysis/SparsityLattice.h"

#include "mlir/IR/Value.h"

namespace proteus {
struct SeedPass {
  static mlir::LogicalResult run(mlir::Block *block, LatticeMap &state);
};

struct ForwardPass {
  static mlir::LogicalResult run(mlir::Block *block, LatticeMap &state);
};

struct LateralPass {
  static mlir::LogicalResult run(mlir::Block *block, LatticeMap &state);
};

struct BackwardPass {
  static mlir::LogicalResult run(mlir::Block *block, LatticeMap &state);
};

class SparsityAnalysis {
public:
  mlir::LogicalResult run(mlir::Block *block);
  const LatticeMap &getState() const;
  LatticeMap &getState();
  SparsityLattice *getState(const mlir::Value &value);
  const SparsityLattice *getState(const mlir::Value &value) const;

private:
  template <typename PassType> mlir::LogicalResult run(mlir::Block *block);

  LatticeMap state;
};
} // namespace proteus
