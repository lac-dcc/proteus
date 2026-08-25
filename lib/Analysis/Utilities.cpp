#include "Analysis/Utilities.h"

#include "Analysis/SeedPass.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

void proteus::printState(
    const llvm::DenseMap<mlir::Value, SparsityLattice> &state) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  os << "=== SPA State Table ===\n";
  for (const auto &[val, lattice] : state) {
    os << "  ";
    val.print(os);
    if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(val)) {
      os << " (%arg" << ba.getArgNumber() << ")";
    }
    os << "\n  \xE2\x86\xB3 ";
    std::string latBuf;
    llvm::raw_string_ostream latOs(latBuf);
    latOs << lattice;
    latOs.flush();
    for (char c : latBuf) {
      os << (c == '\n' ? "\n      " : llvm::StringRef(&c, 1));
    }
    os << "\n";
  }
  os << "======================\n";
  llvm::outs() << os.str();
}

proteus::SparsityLattice proteus::observeMemref(DynamicMemRefType<float> mref) {
  llvm::SmallVector<uint64_t> shape(mref.sizes, mref.sizes + mref.rank);
  SparsityLattice lattice(shape);

  for (int64_t d = 0; d < mref.rank; ++d) {
    lattice[d].reset();
  }

  if (mref.rank == 0) {
    return lattice;
  }

  llvm::SmallVector<uint64_t> shapeStrides(mref.rank);
  shapeStrides[mref.rank - 1] = 1;
  for (int64_t d = mref.rank - 2; d >= 0; --d) {
    shapeStrides[d] = shapeStrides[d + 1] * shape[d + 1];
  }

  uint64_t total = shapeStrides[0] * shape[0];
  for (uint64_t index = 0; index < total; ++index) {
    uint64_t remaining = index;
    int64_t offset = mref.offset;
    for (int64_t d = 0; d < mref.rank; ++d) {
      uint64_t sliceIdx = remaining / shapeStrides[d];
      remaining %= shapeStrides[d];
      offset += static_cast<int64_t>(sliceIdx) * mref.strides[d];
    }
    if (mref.data[offset] != 0.0F) {
      SeedPass::markSlices(index, shapeStrides, lattice);
    }
  }

  return lattice;
}

std::pair<uint64_t, uint64_t>
proteus::ZeroCounter::count(const SparsityLattice &lattice) {
  uint64_t zeros = 0;
  uint64_t totals = 0;

  for (uint64_t dim = 0; dim < lattice.rank(); ++dim) {
    const llvm::BitVector &bv = lattice[dim];
    zeros += bv.size() - bv.count();
    totals += bv.size();
  }

  return std::pair{zeros, totals};
}

ZeroMap proteus::ZeroCounter::count(const LatticeMap &state) {
  ZeroMap result;

  result.reserve(state.size());
  for (const auto &[val, lattice] : state) {
    result[val] = count(lattice);
  }

  return result;
}

void proteus::ZeroCounter::print(const LatticeMap &state,
                                 llvm::raw_ostream &os) {
  uint64_t grandTotal = 0;
  uint64_t absoluteTotal = 0;

  auto latticeTotals = count(state);

  os << "=== Zero Counts ===\n";
  for (const auto &[val, lattice] : state) {
    os << "  [";
    if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(val)) {
      os << "block arg #" << ba.getArgNumber();
    } else if (mlir::Operation *op = val.getDefiningOp()) {
      auto result = llvm::cast<mlir::OpResult>(val);
      os << op->getName().getStringRef();
      if (op->getNumResults() > 1) {
        os << " result #" << result.getResultNumber();
      }
    }
    os << "]\n  \xE2\x86\xB3 ";

    for (uint64_t dim = 0; dim < lattice.rank(); ++dim) {
      const llvm::BitVector &bv = lattice[dim];
      uint64_t zeros = bv.size() - bv.count();
      if (dim > 0) {
        os << "  ";
      }

      os << "dim[" << dim << "]: " << zeros << "/" << bv.size() << " zeros";
    }

    os << "  | total: " << latticeTotals[val].first << "\n";
    grandTotal += latticeTotals[val].first;
    absoluteTotal += latticeTotals[val].second;
  }
  os << "Grand total: " << grandTotal << " zero(s)\n";
  os << "Absolute total: " << absoluteTotal << " zero(s)\n";
  os << "===================\n";
}
