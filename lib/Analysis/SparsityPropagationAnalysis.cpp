#include "Analysis/SparsityPropagationAnalysis.h"

#include "Analysis/SparsityLattice.h"
#include "Analysis/Utilities.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Timing.h"

namespace proteus {

std::optional<PassStage>
SparsityPropagationAnalysis::getPassStage(llvm::StringRef name) {
  if (name == "seed") {
    return PassStage::Seed;
  }
  if (name == "forward") {
    return PassStage::Forward;
  }
  if (name == "lateral") {
    return PassStage::Lateral;
  }
  if (name == "backward") {
    return PassStage::Backward;
  }

  return std::nullopt;
}

void SparsityPropagationAnalysis::runOnOperation() {

  auto stage = getPassStage(passStage);

  if (!stage) {
    getOperation().emitError()
        << "spa-analysis: invalid last-pass value '" << passStage
        << "'; expected one of: seed, forward, lateral, backward";
    signalPassFailure();
    return;
  }

  if (!seedLattice.empty()) {
    if (getOperation().getNumArguments() == 0) {
      getOperation().emitError()
          << "spa-analysis: seed-lattice given but function has no arguments";
      signalPassFailure();
      return;
    }

    auto attr = mlir::parseAttribute(seedLattice, getOperation().getContext());
    auto arrayAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr);
    if (!arrayAttr) {
      getOperation().emitError()
          << "spa-analysis: seed-lattice must parse to an ArrayAttr: '"
          << seedLattice << "'";
      signalPassFailure();
      return;
    }

    getOperation().setArgAttr(0, "proteus.lattice", arrayAttr);
  }

  std::unique_ptr<mlir::DefaultTimingManager> tm;
  if (timePasses) {
    tm = std::make_unique<mlir::DefaultTimingManager>();
    tm->setEnabled(true);
  }
  mlir::TimingScope rootScope = tm ? tm->getRootScope() : mlir::TimingScope();

  SparsityEngine sa;
  for (auto &block : getOperation().getBody()) {
    sa.run(&block, *stage, tm ? &rootScope : nullptr);
  }

  getOperation().walk([&](mlir::Operation *op) {
    if (op->getNumResults() == 1) {
      auto *lattice = sa.getState(op->getResult(0));

      if (latticeDump && lattice) {
        op->setAttr("proteus.lattice",
                    SparsityLattice::toAttr(*lattice, op->getContext()));
      }
    }
  });

  if (latticeDump) {
    for (auto arg : getOperation().getArguments()) {
      auto *lattice = sa.getState(arg);
      if (lattice != nullptr) {
        getOperation().setArgAttr(
            arg.getArgNumber(), "proteus.lattice",
            SparsityLattice::toAttr(*lattice, getOperation().getContext()));
      }
    }
  }

  if (stateDump) {
    printState(sa.getState());
  }

  if (printZeros) {
    proteus::ZeroCounter::print(sa.getState());
  }
}

std::unique_ptr<mlir::Pass> createSparsityPropagationAnalysis() {
  return std::make_unique<SparsityPropagationAnalysis>();
}
} // namespace proteus
