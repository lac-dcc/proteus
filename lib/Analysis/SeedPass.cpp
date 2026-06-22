#include "Analysis/SeedPass.h"

#include "Analysis/SparsityEngine.h"
#include "Analysis/SparsityLattice.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

static std::optional<proteus::SparsityLattice>
resolveArgLattice(mlir::BlockArgument arg, mlir::func::FuncOp funcOp) {
  auto dict = funcOp.getArgAttrDict(arg.getArgNumber());
  if (dict) {
    if (auto lattice = proteus::SparsityLattice::fromAttr(dict)) {
      return lattice;
    }
  }
  return proteus::SparsityLattice::defaultFromValue(arg);
}

Result proteus::SeedPass::run(mlir::Block *block, SparsityEngine &analysis) {
  if (!block->isEntryBlock()) {
    return mlir::success();
  }

  auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(block->getParentOp());
  if (!funcOp) {
    return mlir::success();
  }

  for (auto &arg : block->getArguments()) {
    auto lattice = resolveArgLattice(arg, funcOp);
    if (lattice.has_value()) {
      analysis.getState().try_emplace(arg, lattice.value());
    }
  }

  return mlir::success();
}

void proteus::SeedPass::markSlices(uint64_t index,
                                   llvm::SmallVector<uint64_t> &strides,
                                   SparsityLattice &lattice) {
  uint64_t remaining = index;
  for (std::size_t d = 0; d < lattice.rank(); ++d) {
    uint64_t sliceIdx = remaining / strides[d];
    remaining %= strides[d];
    lattice[d].set(sliceIdx);
  }
}

void proteus::SeedPass::seedSplat(mlir::DenseElementsAttr &denseAttr,
                                  bool isFloat, SparsityLattice &lattice) {
  bool isZero = isFloat ? denseAttr.getSplatValue<llvm::APFloat>().isZero()
                        : denseAttr.getSplatValue<llvm::APInt>().isZero();
  if (isZero) {
    for (int64_t d = 0; d < static_cast<int64_t>(lattice.rank()); ++d) {
      lattice[d].reset();
    }
  }
}

void proteus::SeedPass::seedNonSplat(mlir::DenseElementsAttr &denseAttr,
                                     bool isFloat, SparsityLattice &lattice) {

  for (std::size_t d = 0; d < lattice.rank(); ++d) {
    lattice[d].reset();
  }

  if (lattice.rank() == 0) {
    return;
  }

  llvm::SmallVector<uint64_t> strides(lattice.rank());
  strides[lattice.rank() - 1] = 1;
  for (int64_t d = static_cast<int64_t>(lattice.rank()) - 2; d >= 0; --d) {
    strides[d] = strides[d + 1] * lattice[d + 1].size();
  }

  if (isFloat) {
    uint64_t index = 0;
    for (auto val : denseAttr.getValues<llvm::APFloat>()) {
      if (!val.isZero()) {
        markSlices(index, strides, lattice);
      }
      ++index;
    }
  } else {
    int64_t index = 0;
    for (auto val : denseAttr.getValues<llvm::APInt>()) {
      if (!val.isZero()) {
        markSlices(index, strides, lattice);
      }
      ++index;
    }
  }
}

void proteus::SeedPass::seedConstant(mlir::arith::ConstantOp op,
                                     SparsityEngine &analysis) {

  auto lattice = SparsityLattice::defaultFromValue(op.getResult());
  if (!lattice.has_value()) {
    return;
  }

  auto tensorType = mlir::cast<mlir::RankedTensorType>(op.getType());
  auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
  if (!denseAttr) {
    analysis.getState().try_emplace(op.getResult(), lattice.value());
    return;
  }

  auto elemType = tensorType.getElementType();
  bool isFloat = mlir::isa<mlir::FloatType>(elemType);
  bool isInt = mlir::isa<mlir::IntegerType>(elemType);
  if (!isFloat && !isInt) {
    analysis.getState().try_emplace(op.getResult(), lattice.value());
    return;
  }

  if (denseAttr.isSplat()) {
    seedSplat(denseAttr, isFloat, lattice.value());
  } else {
    seedNonSplat(denseAttr, isFloat, lattice.value());
  }

  analysis.getState().try_emplace(op.getResult(), lattice.value());
}
