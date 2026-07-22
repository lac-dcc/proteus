#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <utility>

namespace proteus {

class SparsityLattice;

using LatticeMap = llvm::DenseMap<mlir::Value, SparsityLattice>;

/**
 * @brief Sparsity lattice for a ranked tensor.
 *
 * Represents the sparsity of a ranked tensor as a collection of BitVectors,
 * one per dimension. Each bit corresponds to a slice along that dimension:
 * a set bit means the slice is dense, a cleared bit means it is sparse.
 *
 * The lattice is initialised fully dense (all bits set). Analysis passes
 * clear bits as they prove sparsity. Dimension ordering matches the tensor
 * shape: index 0 is the outermost dimension, index rank-1 is the innermost.
 */
class SparsityLattice {
public:
  /**
   * @brief Constructs a fully-dense lattice for a tensor with the given shape.
   *
   * One BitVector of size shape[i] is created per dimension, with all bits
   * initialised to true (dense).
   *
   * @param shape The size of each tensor dimension.
   */
  SparsityLattice(llvm::ArrayRef<uint64_t> shape);

  /**
   * @brief Returns the number of dimensions (rank) of the lattice.
   * @return Number of BitVectors stored in the lattice.
   */
  [[nodiscard]] uint64_t rank() const;

  /**
   * @brief Returns the size of each dimension.
   * @return A vector of dimension sizes matching the shape passed to the
   *         constructor.
   */
  [[nodiscard]] llvm::SmallVector<uint64_t> shape() const;

  /**
   * @brief Returns a mutable reference to the BitVector for the given
   * dimension.
   * @param index Zero-based dimension index.
   * @return Reference to the BitVector for dimension @p index.
   */
  llvm::BitVector &operator[](uint64_t index);

  /**
   * @brief Returns a read-only reference to the BitVector for the given
   * dimension.
   * @param index Zero-based dimension index.
   * @return Const reference to the BitVector for dimension @p index.
   */
  const llvm::BitVector &operator[](uint64_t index) const;

  /**
   * @brief Compares two lattices for equality.
   *
   * Two lattices are equal if they have the same rank and identical BitVectors
   * in every dimension.
   *
   * @param other The lattice to compare against.
   * @return True if both lattices are identical, false otherwise.
   */
  bool operator==(const SparsityLattice &other) const;

  /**
   * @brief Compares two lattices for inequality.
   *
   * Two lattices are non equal if they have different rank and different
   * BitVectors in every dimension.
   *
   * @param other The lattice to compare against.
   * @return True if lattices are different, false otherwise.
   */
  bool operator!=(const SparsityLattice &other) const;

  /**
   * @brief Computes the least-upper-bound (join) of two lattices.
   *
   * A bit is set (dense) in the result if it is set in either @p a or @p b.
   *
   * @param a First lattice operand.
   * @param b Second lattice operand.
   * @pre a.shape() == b.shape()
   * @return A new lattice representing the least-upper-bound.
   */
  static SparsityLattice join(const SparsityLattice &a,
                              const SparsityLattice &b);

  /**
   * @brief Computes the greatest-lower-bound (meet) of two lattices.
   *
   * A bit is set (dense) in the result only if it is set in both @p a and
   * @p b; it is cleared (sparse) if either operand is sparse.   *
   *
   * @param a First lattice operand.
   * @param b Second lattice operand.
   * @pre a.shape() == b.shape()
   * @return A new lattice representing the greatest-lower-bound.
   */
  static SparsityLattice meet(const SparsityLattice &a,
                              const SparsityLattice &b);

  /**
   * @brief Returns every contiguous range of set bits.
   *
   * Scans the bitvector for consecutive set bits and arranges them to (offset,
   * size) pairs
   *
   * @param bv The BitVector to scan
   * @return A vector of (offset, size) pairs, one per dense range.
   */
  static llvm::SmallVector<std::pair<int64_t, int64_t>>
  getDensityRanges(const llvm::BitVector &bv);

  /**
   * @brief Serialises the lattice to an MLIR ArrayAttr for IR annotation.
   *
   * Each dimension is encoded as a DictionaryAttr with two entries:
   * - "size": i64 storing the number of bits.
   * - "words": DenseI64Array storing the packed BitVector words.
   *
   * @param lattice The lattice to serialise.
   * @param ctx     The MLIR context used to create attributes.
   * @return An ArrayAttr containing one DictionaryAttr per dimension.
   */
  static mlir::ArrayAttr toAttr(const SparsityLattice &lattice,
                                mlir::MLIRContext *ctx);

  /**
   * @brief Prints the lattice using the same `{size = .., words = array<i64:
   * ..>}` syntax toAttr serialises to an MLIR attribute, without requiring an
   * mlir::MLIRContext.
   *
   * @param lattice The lattice to print.
   * @param os      The output stream to write to.
   */
  static void printAsAttr(const SparsityLattice &lattice,
                          llvm::raw_ostream &os);

  /**
   * @brief Deserialises the MLIR dictionary attribute to a SparsityLattice
   * object.
   *
   * Looks for a "proteus.sparsity" key in @p dict. Returns std::nullopt if the
   * key is absent, allowing callers to defer to the forward pass.
   *
   * @param dict A DictionaryAttr on a function argument.
   * @return The reconstructed lattice, or std::nullopt if no sparsity
   * annotation is present.
   */
  static std::optional<SparsityLattice>
  fromAttr(const mlir::DictionaryAttr &dict);

  /**
   * @brief Deserialises the MLIR array attribute to a SparsityLattice
   * object.
   *
   * Looks for a "proteus.sparsity" key in @p dict. Returns std::nullopt if the
   * key is absent, allowing callers to defer to the forward pass.
   *
   * @param dict A ArrayAttr on a function argument.
   * @return The reconstructed lattice, or std::nullopt if no sparsity
   * annotation is present.
   */
  static std::optional<SparsityLattice>
  fromAttr(const mlir::ArrayAttr &arrayAttr);

  /**
   * @brief Creates a lattice from a ranked tensor value
   *
   * The returned lattice is fully dense. Returns std::nullopt if the operation
   * has no ranked tensor result with a static shape.
   *
   * @param value The MLIR value to inspect.
   * @return A SparsityLattice, or nullopt.
   */
  static std::optional<SparsityLattice>
  defaultFromValue(const mlir::Value &value);

  /**
   * @brief Human readable pretty printer for sparsity lattice
   *
   * @param &out The outstream to print out
   * @param &lattice The lattice to print
   * @return The outstream that was altered with the lattice
   */
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &out,
                                       const SparsityLattice &lattice);

private:
  llvm::SmallVector<llvm::BitVector> sparsities;

  /**
   * @brief Helper function for the fromAttr function, for more details refer
   * to the overloaded fromAttr function
   */
  static SparsityLattice constructFromAttr(const mlir::ArrayAttr &arrayAttr);

  /**
   * @brief Takes a reference to a single bitvector of a sparsity lattice
   * and packs into words for printing and attaching as an attribute
   *
   * @param The bitvector to pack into words
   * @return A vector of words
   */
  static llvm::SmallVector<int64_t> packWords(const llvm::BitVector &bv);
};

} // namespace proteus
