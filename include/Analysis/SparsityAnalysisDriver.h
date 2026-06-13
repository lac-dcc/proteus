#include "Analysis/SparsityLattice.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"

using Result = mlir::LogicalResult;

namespace proteus {

struct SeedPass {
  static Result run(mlir::Block *block, LatticeMap &state);
};

struct ForwardPass {
  static Result run(mlir::Block *block, LatticeMap &state);
};

struct LateralPass {
  static Result run(mlir::Block *block, LatticeMap &state);
};

struct BackwardPass {
  static Result run(mlir::Block *block, LatticeMap &state);
};

class SparsityAnalysis {
public:
  mlir::LogicalResult run(mlir::Block *block);
  const LatticeMap &getState() const;
  LatticeMap &getState();
  SparsityLattice *getState(const mlir::Value &value);
  const SparsityLattice *getState(const mlir::Value &value) const;

private:
  template <typename PassType> Result run(mlir::Block *block);
  Result visit(mlir::Operation *op);

  // Linalg visitors
  Result visitOp(mlir::linalg::MatmulOp op);
  // Result visitOp(mlir::linalg::ManyOtherOps op);

  LatticeMap state;
};
} // namespace proteus
