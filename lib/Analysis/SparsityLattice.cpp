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

std::optional<SparsityLattice>
SparsityLattice::fromAttr(const mlir::DictionaryAttr &dict) {
  auto sparsityAttr = dict.get("proteus.lattice");

  if (!sparsityAttr)
    return std::nullopt;

  if (llvm::isa<mlir::ArrayAttr>(sparsityAttr)) {
    auto arrayAttr = llvm::cast<mlir::ArrayAttr>(sparsityAttr);

    llvm::SmallVector<uint64_t> shape;
    for (auto dimAttr : arrayAttr) {
      auto dimDict = llvm::cast<mlir::DictionaryAttr>(dimAttr);
      shape.push_back(static_cast<uint64_t>(
          llvm::cast<mlir::IntegerAttr>(dimDict.get("size")).getInt()));
    }

    return constructFromAttr(arrayAttr, shape);
  }

  return std::nullopt;
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

std::optional<SparsityLattice>
SparsityLattice::defaultFromValue(const mlir::Value &value) {
  auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return std::nullopt;

  auto shape = tensorType.getShape();
  llvm::SmallVector<uint64_t> uShape(shape.begin(), shape.end());
  auto lattice = SparsityLattice(uShape);

  return lattice;
}

SparsityLattice
SparsityLattice::constructFromAttr(const mlir::ArrayAttr &arrayAttr,
                                   const llvm::SmallVector<uint64_t> &shape) {
  SparsityLattice lattice(shape);
  // TODO: to implement reconstruction here from attribute
  return lattice;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &out,
                              const SparsityLattice &lattice) {
  out << "SparsityLattice<";
  for (uint64_t d = 0; d < lattice.sparsities.size(); ++d) {
    if (d > 0)
      out << "x";
    out << lattice.sparsities[d].size();
  }
  out << "> {\n";
  for (uint64_t d = 0; d < lattice.sparsities.size(); ++d) {
    const auto &bv = lattice.sparsities[d];
    out << "  a[" << d << "]: ";
    for (uint64_t i = 0; i < bv.size(); ++i)
      out << (bv[i] ? '1' : '0');
    out << "  (" << bv.size() << " slices)\n";
  }
  out << "}";
  return out;
}

} // namespace proteus
