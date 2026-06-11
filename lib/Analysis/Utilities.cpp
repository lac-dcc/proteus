#include "Analysis/Utilities.h"

#include "llvm/Support/raw_ostream.h"

void proteus::printState(
    const llvm::DenseMap<mlir::Value, SparsityLattice> &state) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  os << "=== SPA State Table ===\n";
  for (const auto &[val, lattice] : state) {
    os << "  ";
    val.print(os);
    os << " -> " << lattice << "\n";
  }
  os << "======================\n";
  llvm::outs() << os.str();
}
