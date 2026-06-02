#include "Analysis/SparsityLattice.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

namespace proteus {

SparsityLattice::SparsityLattice(llvm::ArrayRef<uint64_t> shape) {
  for (auto s : shape) {
    sparsities.emplace_back(s, true);
  }
}

llvm::BitVector &SparsityLattice::operator[](const uint64_t index) {
  return sparsities[index];
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

std::shared_ptr<SparsityLattice>
SparsityLattice::getSparsityLattice(mlir::Operation *op) {
  return mlir::TypeSwitch<mlir::Operation *, std::shared_ptr<SparsityLattice>>(
             op)
      .Default([](auto) { return nullptr; });
}

} // namespace proteus
