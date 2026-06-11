#include "Analysis/SparsityLattice.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace proteus {
using LatticeMap = llvm::DenseMap<mlir::Value, proteus::SparsityLattice>;

struct ForwardPass {
  static mlir::LogicalResult run(mlir::Block *block, LatticeMap &state);
};

class SparsityAnalysis {
public:
  mlir::LogicalResult run(mlir::Block *block);
  const LatticeMap &getState() const;
  LatticeMap &getState();

private:
  template <typename PassType> mlir::LogicalResult run(mlir::Block *block);

  LatticeMap state;
};
} // namespace proteus
