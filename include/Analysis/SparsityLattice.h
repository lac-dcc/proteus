#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/raw_ostream.h"

namespace proteus {

class SparsityLattice {
public:
  SparsityLattice(llvm::ArrayRef<uint64_t> shape);

  llvm::BitVector &operator[](const uint64_t index);
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &out,
                                       const SparsityLattice &lattice);

  static mlir::ArrayAttr toAttr(const SparsityLattice &lattice,
                                mlir::MLIRContext *ctx);

private:
  llvm::SmallVector<llvm::BitVector> sparsities;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &out,
                                     const SparsityLattice &lattice) {
  out << "SparsityLattice [\n";
  for (const auto &bv : lattice.sparsities) {
    for (uint64_t i = 0; i < bv.size(); i++) {
      out << (bv[i] ? '1' : '0');
    }
  }
  out << "\n";

  return out;
};

} // namespace proteus
