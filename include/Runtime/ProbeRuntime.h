#pragma once

#include "Analysis/SparsityLattice.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include "llvm/ADT/DenseMap.h"
#include <cstdint>

#define PROTEUS_PROBE_EXPORT __attribute__((visibility("default")))

/*
 * @brief Global state of observations per pair of {opID, resultID}
 */
static llvm::DenseMap<std::pair<int32_t, int32_t>, proteus::SparsityLattice>
    state; // NOLINT

/**
 * @brief Records a runtime observation of a tensor's buffer.
 *
 * @param memref   Unranked memref view over the observed tensor's runtime
 * buffer.
 * @param opID     Identifier of the op that produced the observed value.
 * @param resultID Identifier of the op's result being observed.
 */
extern "C" PROTEUS_PROBE_EXPORT void
_mlir_ciface_probeObserveMemrefF32( // NOLINT
    UnrankedMemRefType<float> *memref, int32_t opID, int32_t resultID);

/**
 * @brief Reports observations recorded so far, one per (opID, resultID) pair.
 */
extern "C" PROTEUS_PROBE_EXPORT void _mlir_ciface_probeReport(); // NOLINT
