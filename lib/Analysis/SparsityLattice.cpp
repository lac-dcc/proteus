#include "Analysis/SparsityLattice.h"

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
  if (sparsities.size() != other.sparsities.size()) {
    return false;
  }
  for (size_t i = 0; i < sparsities.size(); ++i) {
    if (sparsities[i] != other.sparsities[i]) {
      return false;
    }
  }
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

SparsityLattice SparsityLattice::meet(const SparsityLattice &a,
                                      const SparsityLattice &b) {
  if (a.shape() != b.shape()) {
    llvm::report_fatal_error("Shapes of lattices when meeting are different.");
  }

  SparsityLattice lattice(a.shape());

  for (std::size_t i = 0; i < lattice.rank(); i++) {
    lattice[i] = a[i];
    lattice[i] &= b[i];
  }

  return lattice;
}

llvm::SmallVector<std::pair<int64_t, int64_t>>
SparsityLattice::getDensityRanges(const llvm::BitVector &bv) {
  llvm::SmallVector<std::pair<int64_t, int64_t>> ranges;
  auto size = static_cast<int64_t>(bv.size());

  int64_t start = -1;
  // Traverse through the bitvector
  for (int64_t i = 0; i < size; ++i) {
    // If the bit is set
    if (bv[i]) {
      // We should either begin a contiguous range
      // Or just iterate through all the set bits in the current range
      if (start == -1) {
        start = i;
      }
      // If the bit is not set and we just got out of a contiguous dense range
    } else if (start != -1) {
      // We need to insert the range into the result vector
      ranges.emplace_back(start, i - start);
      // And default our start value for the next range
      start = -1;
    }
  }

  // In the case where the last item in the range is
  // set that constitutes a range at the end of the bitvector so add that as
  // well
  if (start != -1) {
    ranges.emplace_back(start, size - start);
  }

  return ranges;
}

llvm::SmallVector<int64_t>
SparsityLattice::packWords(const llvm::BitVector &bv) {
  llvm::SmallVector<int64_t> words;
  for (unsigned i = 0; i < bv.size(); i += 64) {
    int64_t word = 0;
    for (unsigned b = i; b < std::min(i + 64U, bv.size()); ++b) {
      if (bv[b]) {
        word |= (int64_t{1} << (b - i));
      }
    }
    words.push_back(word);
  }

  return words;
}

mlir::ArrayAttr SparsityLattice::toAttr(const SparsityLattice &lattice,
                                        mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::Attribute> bvAttrs;
  for (const auto &bv : lattice.sparsities) {
    llvm::SmallVector<int64_t> words = packWords(bv);
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

void SparsityLattice::printAsAttr(const SparsityLattice &lattice,
                                  llvm::raw_ostream &os) {
  os << "[";
  for (uint64_t d = 0; d < lattice.sparsities.size(); ++d) {
    if (d > 0) {
      os << ", ";
    }
    const llvm::BitVector &bv = lattice.sparsities[d];
    llvm::SmallVector<int64_t> words = packWords(bv);
    os << "{size = " << bv.size() << " : i64, words = array<i64:";
    for (size_t w = 0; w < words.size(); ++w) {
      os << (w == 0 ? " " : ", ") << words[w];
    }
    os << ">}";
  }
  os << "]";
}

std::optional<SparsityLattice>
SparsityLattice::fromAttr(const mlir::ArrayAttr &arrayAttr) {
  return constructFromAttr(arrayAttr);
}

std::optional<SparsityLattice>
SparsityLattice::fromAttr(const mlir::DictionaryAttr &dict) {
  auto sparsityAttr = dict.get("proteus.lattice");

  if (!sparsityAttr) {
    return std::nullopt;
  }

  if (llvm::isa<mlir::ArrayAttr>(sparsityAttr)) {
    auto arrayAttr = llvm::cast<mlir::ArrayAttr>(sparsityAttr);
    return constructFromAttr(arrayAttr);
  }

  return std::nullopt;
}

std::optional<SparsityLattice>
SparsityLattice::defaultFromValue(const mlir::Value &value) {
  auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorType || !tensorType.hasStaticShape()) {
    return std::nullopt;
  }

  auto shape = tensorType.getShape();
  llvm::SmallVector<uint64_t> uShape(shape.begin(), shape.end());
  auto lattice = SparsityLattice(uShape);

  return lattice;
}

SparsityLattice
SparsityLattice::constructFromAttr(const mlir::ArrayAttr &arrayAttr) {
  // Setup the shape of the lattice initially
  llvm::SmallVector<uint64_t> shape(arrayAttr.size());

  // For each item within the proteus lattice attribute,
  // we will assign a bitvector
  for (auto [i, dimAttr] : llvm::enumerate(arrayAttr)) {
    auto dimDict = llvm::cast<mlir::DictionaryAttr>(dimAttr);
    auto dimSize = static_cast<uint64_t>(
        llvm::cast<mlir::IntegerAttr>(dimDict.get("size")).getInt());
    shape[i] = dimSize;
  }

  // Construct the lattice with the above shape derived from the
  // attribute
  SparsityLattice lattice(shape);

  // For each bitvector we need to see which bits we need to set
  for (size_t i = 0; i < arrayAttr.size(); ++i) {
    auto dimDict = llvm::cast<mlir::DictionaryAttr>(arrayAttr[i]);
    uint64_t size = static_cast<uint64_t>(
        llvm::cast<mlir::IntegerAttr>(dimDict.get("size")).getInt());
    auto words = llvm::cast<mlir::DenseI64ArrayAttr>(dimDict.get("words"));

    // Grab the bitvector for the default lattice
    auto &bv = lattice[i];
    // And reset it because the default behaviour is for the bitvector
    // to be all true
    bv.reset();

    // Iterate through each word in the attribute
    for (size_t w = 0; w < static_cast<size_t>(words.size()); ++w) {
      auto word = static_cast<uint64_t>(words[w]);
      // w * 64 + b represents the global offset in the case where there's
      // multiple words in a single bitvector, meaning the bitvector has
      // length larger than 64. So the conditional in the `if` statement
      // becomes twofold, not exceeding 64 bits for a single word and
      // not exceeding global size for the entire bitvector
      for (unsigned b = 0; b < 64 && (w * 64) + b < size; ++b) {
        // If the bit b is set in the word, then set that bit in the bitvector
        if (((word >> b) & 1U) != 0U) {
          bv.set((w * 64) + b);
        }
      }
    }
  }

  return lattice;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &out,
                              const SparsityLattice &lattice) {
  out << "SparsityLattice<";
  for (uint64_t d = 0; d < lattice.sparsities.size(); ++d) {
    if (d > 0) {
      out << "x";
    }
    out << lattice.sparsities[d].size();
  }
  out << "> {\n";
  for (uint64_t d = 0; d < lattice.sparsities.size(); ++d) {
    const auto &bv = lattice.sparsities[d];
    out << "  a[" << d << "]: ";
    for (uint64_t i = 0; i < bv.size(); ++i) {
      out << (bv[i] ? '1' : '0');
    }
    out << "  (" << bv.size() << " slices)\n";
  }
  out << "}";
  return out;
}

} // namespace proteus
