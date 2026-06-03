#include "Analysis/SparsityLattice.h"

#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

TEST(SparsityLattice, ConstructionSetsAllDense) {
  proteus::SparsityLattice lat({80, 512});
  EXPECT_EQ(lat[0].size(), 80u);
  EXPECT_EQ(lat[1].size(), 512u);
  EXPECT_TRUE(lat[0].all());
  EXPECT_TRUE(lat[1].all());
}

TEST(SparsityLattice, IndexMutatesCorrectDimension) {
  proteus::SparsityLattice lat({80, 512});
  lat[0].reset(5);
  EXPECT_FALSE(lat[0][5]);
  EXPECT_TRUE(lat[1].all());
}

TEST(SparsityLattice, ToAttrWordCounts) {
  mlir::MLIRContext ctx;
  proteus::SparsityLattice lat({80, 512});
  auto attr = proteus::SparsityLattice::toAttr(lat, &ctx);

  EXPECT_EQ(attr.size(), 2);

  auto bv0 = llvm::cast<mlir::DictionaryAttr>(attr[0]);
  auto bv1 = llvm::cast<mlir::DictionaryAttr>(attr[1]);

  auto bv0_words = llvm::cast<mlir::DenseI64ArrayAttr>(bv0.get("words"));
  auto bv1_words = llvm::cast<mlir::DenseI64ArrayAttr>(bv1.get("words"));

  EXPECT_EQ(bv0_words.size(), 2);
  EXPECT_EQ(bv1_words.size(), 8);
  EXPECT_EQ(bv0_words[0], -1);

  auto size0 = llvm::cast<mlir::IntegerAttr>(bv0.get("size")).getInt();
  auto size1 = llvm::cast<mlir::IntegerAttr>(bv1.get("size")).getInt();

  EXPECT_EQ(size0, 80);
  EXPECT_EQ(size1, 512);
}

TEST(SparsityLattice, RankMatchesNumberOfDimensions) {
  proteus::SparsityLattice lat({4, 8, 16});
  EXPECT_EQ(lat.rank(), 3u);
}

TEST(SparsityLattice, ShapeMatchesConstructorInput) {
  proteus::SparsityLattice lat({4, 8, 16});
  auto s = lat.shape();
  ASSERT_EQ(s.size(), 3u);
  EXPECT_EQ(s[0], 4u);
  EXPECT_EQ(s[1], 8u);
  EXPECT_EQ(s[2], 16u);
}

TEST(SparsityLattice, EqualityHoldsForIdenticalLattices) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
  EXPECT_EQ(a, b);
}

TEST(SparsityLattice, EqualityFailsWhenBitsDiffer) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
  b[0].reset(2);
  EXPECT_NE(a, b);
}

TEST(SparsityLattice, EqualityFailsWhenRankDiffers) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8, 16});
  EXPECT_NE(a, b);
}

TEST(SparsityLattice, JoinOfTwoDenseLatticesIsDense) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
  auto result = proteus::SparsityLattice::join(a, b);
  EXPECT_TRUE(result[0].all());
  EXPECT_TRUE(result[1].all());
}

TEST(SparsityLattice, JoinPreservesSparseBitsAgreedByBothOperands) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
  // Both agree that row 1 is sparse.
  a[0].reset(1);
  b[0].reset(1);
  auto result = proteus::SparsityLattice::join(a, b);
  EXPECT_FALSE(result[0][1]);
}

TEST(SparsityLattice, JoinDropsSparseBitNotAgreedByBothOperands) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
  // Only a considers row 2 sparse; b does not.
  a[0].reset(2);
  auto result = proteus::SparsityLattice::join(a, b);
  EXPECT_EQ(result, b);
}

TEST(SparsityLattice, JoinIsCommutative) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
  a[0].reset(0);
  b[1].reset(3);
  auto ab = proteus::SparsityLattice::join(a, b);
  auto ba = proteus::SparsityLattice::join(b, a);
  EXPECT_EQ(ab, ba);
}

TEST(SparsityLattice, JoinIsIdempotent) {
  proteus::SparsityLattice a({4, 8});
  a[0].reset(1);
  a[1].reset(5);
  auto result = proteus::SparsityLattice::join(a, a);
  EXPECT_EQ(result, a);
}
