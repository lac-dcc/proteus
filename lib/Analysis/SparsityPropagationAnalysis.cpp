#include "Analysis/SparsityPropagationAnalysis.h"

#include "Analysis/SparsityAnalysis.h"
#include "Analysis/SparsityLattice.h"
#include "Analysis/Utilities.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace proteus {
void SparsityPropagationAnalysis::runOnOperation() {

  SparsityAnalysis SA;

  for (auto &block : getOperation().getBody())
    if (SA.run(&block).failed())
      return;

  getOperation().walk([&](mlir::Operation *op) {
    for (auto value : op->getOperands()) {
      // Get the operand value, make sure it's a ranked tensor type
      auto operand = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
      // Check if it exists or is an nullptr
      if (!operand || !operand.hasStaticShape())
        return;

      // Get the value's shape
      auto shape = operand.getShape();
      // Create the shape of the lattice
      llvm::SmallVector<uint64_t> s(shape.begin(), shape.end());
      // Create a default all dense lattice
      SparsityLattice lattice(s);

      // Add the value to the block's lattice map
      SA.getState().try_emplace(value, lattice);

      if (latticeDump)
        op->setAttr("proteus.lattice",
                    SparsityLattice::toAttr(lattice, op->getContext()));
    }

    if (op->getNumResults() == 1) {
      // Get the result value, make sure it's a ranked tensor type
      auto result =
          llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
      // Check if it exists or is an nullptr
      if (!result || !result.hasStaticShape())
        return;

      // Get the result's shape
      auto shape = result.getShape();
      // Create the shape of the lattice
      llvm::SmallVector<uint64_t> s(shape.begin(), shape.end());
      // Create a default all dense lattice
      SparsityLattice lattice(s);

      if (latticeDump)
        op->setAttr("proteus.lattice",
                    SparsityLattice::toAttr(lattice, op->getContext()));
    }
  });

  if (stateDump)
    printState(SA.getState());
}

std::unique_ptr<mlir::Pass> createSparsityPropagationAnalysis() {
  return std::make_unique<SparsityPropagationAnalysis>();
}
} // namespace proteus
