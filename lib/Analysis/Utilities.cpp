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
    if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(val))
      os << " (%arg" << ba.getArgNumber() << ")";
    os << "\n  ╰─▶ ";
    std::string latBuf;
    llvm::raw_string_ostream latOs(latBuf);
    latOs << lattice;
    latOs.flush();
    for (char c : latBuf)
      os << (c == '\n' ? "\n      " : llvm::StringRef(&c, 1));
    os << "\n";
  }
  os << "======================\n";
  llvm::outs() << os.str();
}
