#include "Analysis/SparsityLattice.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace proteus {

SparsityLattice::SparsityLattice(llvm::ArrayRef<uint64_t> shape) {
  for (auto s : shape) {
    sparsities.emplace_back(s, true);
  }
}

uint64_t SparsityLattice::rank() const { return sparsities.size(); }

llvm::SmallVector<uint64_t> SparsityLattice::shape() const {
  llvm::SmallVector<uint64_t> shapes;
  for (const auto &bv : sparsities) {
    shapes.push_back(bv.size());
  }
  return shapes;
}

llvm::BitVector &SparsityLattice::operator[](const uint64_t index) {
  return sparsities[index];
}

const llvm::BitVector &SparsityLattice::operator[](const uint64_t index) const {
  return sparsities[index];
}

bool SparsityLattice::operator==(const SparsityLattice &other) const {
  if (sparsities.size() != other.sparsities.size())
    return false;
  for (size_t i = 0; i < sparsities.size(); ++i)
    if (sparsities[i] != other.sparsities[i])
      return false;
  return true;
}

bool SparsityLattice::operator!=(const SparsityLattice &other) const {
  return (!(*this == other));
}

SparsityLattice SparsityLattice::join(const SparsityLattice &a,
                                      const SparsityLattice &b) {
  if (a.shape() != b.shape()) {
    llvm::report_fatal_error("Shapes of lattices when joining are different.");
  }

  SparsityLattice lattice(a.shape());

  for (std::size_t i = 0; i < lattice.rank(); i++) {
    lattice[i] = a[i];
    lattice[i] |= b[i];
  }

  return lattice;
}

mlir::ArrayAttr SparsityLattice::toAttr(const SparsityLattice &lattice,
                                        mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::Attribute> bvAttrs;
  for (const auto &bv : lattice.sparsities) {
    llvm::SmallVector<int64_t> words;
    for (unsigned i = 0; i < bv.size(); i += 64) {
      int64_t word = 0;
      for (unsigned b = i; b < std::min(i + 64u, bv.size()); ++b)
        if (bv[b])
          word |= (int64_t{1} << (b - i));
      words.push_back(word);
    }
    auto dict = mlir::DictionaryAttr::get(
        ctx, {
                 {mlir::StringAttr::get(ctx, "size"),
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                         bv.size())},
                 {mlir::StringAttr::get(ctx, "words"),
                  mlir::DenseI64ArrayAttr::get(ctx, words)},
             });
    bvAttrs.push_back(dict);
  }
  return mlir::ArrayAttr::get(ctx, bvAttrs);
}

SparsityLattice *SparsityLattice::fromOp(mlir::Operation *op) {
  for (auto result : op->getResults()) {
    auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!tensorType || !tensorType.hasStaticShape())
      continue;
    auto shape = tensorType.getShape();
    llvm::SmallVector<uint64_t> uShape(shape.begin(), shape.end());
    return new SparsityLattice(uShape);
  }
  return nullptr;
}

} // namespace proteus
