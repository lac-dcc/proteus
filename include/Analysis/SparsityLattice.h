#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

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
  uint64_t rank() const;

  /**
   * @brief Returns the size of each dimension.
   * @return A vector of dimension sizes matching the shape passed to the
   *         constructor.
   */
  llvm::SmallVector<uint64_t> shape() const;

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
   * A bit is set in the result only if it is set in both @p a and @p b.
   * This is a conservative intersection: a slice is only guaranteed sparse
   * if all analysis paths agree it is sparse.
   *
   * @param a First lattice operand.
   * @param b Second lattice operand.
   * @pre a.shape() == b.shape()
   * @return A new lattice representing the least-upper-bound.
   */
  static SparsityLattice join(const SparsityLattice &a,
                              const SparsityLattice &b);

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
   * @brief Deserialises the MLIR dictionary attribute to a SparsityLattice
   * object.
   *
   * Looks for a "spa.sparsity" key in @p dict. Returns std::nullopt if the key
   * is absent, allowing callers to defer to the forward pass.
   *
   * @param dict A DictionaryAttr on a function argument.
   * @return The reconstructed lattice, or std::nullopt if no sparsity
   * annotation is present.
   */
  static std::optional<SparsityLattice>
  fromAttr(const mlir::DictionaryAttr &dict);

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

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &out,
                                       const SparsityLattice &lattice);

private:
  llvm::SmallVector<llvm::BitVector> sparsities;

  static SparsityLattice constructFromAttr(const mlir::ArrayAttr &arrayAttr);
};

} // namespace proteus
