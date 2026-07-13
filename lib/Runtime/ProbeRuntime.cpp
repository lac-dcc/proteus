#include "Runtime/ProbeRuntime.h"

#include <iostream>
#include <utility>

extern "C" void
_mlir_ciface_probeObserveMemrefF32(UnrankedMemRefType<float> *memref,
                                   int32_t opID, int32_t resultID) { // NOLINT
  auto pair = std::make_pair(opID, resultID);
  state[pair]++;
}

extern "C" void _mlir_ciface_probeReport() { // NOLINT
  for (auto &[key, count] : state) {
    std::cout << "opID=" << key.first << " resultID=" << key.second
              << " observation=" << count << "\n";
  }
}
