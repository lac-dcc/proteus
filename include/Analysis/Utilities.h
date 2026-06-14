#pragma once

#include "Analysis/SparsityLattice.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace proteus {
void printState(const llvm::DenseMap<mlir::Value, SparsityLattice> &state);
} // namespace proteus
