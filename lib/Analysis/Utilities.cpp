#include "Analysis/Utilities.h"

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

uint64_t proteus::ZeroCounter::count(const SparsityLattice &lattice) {
  uint64_t zeros = 0;

  for (uint64_t dim = 0; dim < lattice.rank(); ++dim) {
    const llvm::BitVector &bv = lattice[dim];
    zeros += bv.size() - bv.count();
  }

  return zeros;
}

ZeroMap proteus::ZeroCounter::count(const LatticeMap &state) {
  llvm::DenseMap<mlir::Value, uint64_t> result;

  result.reserve(state.size());
  for (const auto &[val, lattice] : state) {
    result[val] = count(lattice);
  }

  return result;
}

void proteus::ZeroCounter::print(const LatticeMap &state,
                                 llvm::raw_ostream &os) {
  uint64_t grandTotal = 0;

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

    os << "  | total: " << latticeTotals[val] << "\n";
    grandTotal += latticeTotals[val];
  }
  os << "Grand total: " << grandTotal << " zero(s)\n";
  os << "===================\n";
}
