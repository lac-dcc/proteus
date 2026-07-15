#include "Runtime/ProbeRuntime.h"

#include "Analysis/Utilities.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <utility>

using proteus::SparsityLattice;

extern "C" void
_mlir_ciface_probeObserveMemrefF32(UnrankedMemRefType<float> *memref,
                                   int32_t opID, int32_t resultID) { // NOLINT
  DynamicMemRefType<float> dynMemref(*memref);
  state.try_emplace(std::make_pair(opID, resultID),
                    proteus::observeMemref(dynMemref));
}

extern "C" void _mlir_ciface_probeReport() { // NOLINT
  for (auto &[key, lattice] : state) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    SparsityLattice::printAsAttr(lattice, os);
    std::cout << "opID=" << key.first << " resultID=" << key.second
              << " runtime_lattice=" << buf << "\n";
  }
}
