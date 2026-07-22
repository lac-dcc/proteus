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

  auto bv0Words = llvm::cast<mlir::DenseI64ArrayAttr>(bv0.get("words"));
  auto bv1Words = llvm::cast<mlir::DenseI64ArrayAttr>(bv1.get("words"));

  EXPECT_EQ(bv0Words.size(), 2);
  EXPECT_EQ(bv1Words.size(), 8);
  EXPECT_EQ(bv0Words[0], -1);

  auto size0 = llvm::cast<mlir::IntegerAttr>(bv0.get("size")).getInt();
  auto size1 = llvm::cast<mlir::IntegerAttr>(bv1.get("size")).getInt();

  EXPECT_EQ(size0, 80);
  EXPECT_EQ(size1, 512);
}

TEST(SparsityLattice, PrintAsAttrMatchesToAttrPrintedText) {
  mlir::MLIRContext ctx;
  proteus::SparsityLattice lat({4, 80});

  auto attr = proteus::SparsityLattice::toAttr(lat, &ctx);
  std::string fromAttr;
  llvm::raw_string_ostream attrOs(fromAttr);
  attr.print(attrOs);

  std::string fromHelp;
  llvm::raw_string_ostream helperOs(fromHelp);
  proteus::SparsityLattice::printAsAttr(lat, helperOs);

  EXPECT_EQ(fromAttr, fromHelp);
}

TEST(SparsityLattice, RankMatchesNumberOfDimensions) {
  proteus::SparsityLattice lat({4, 8, 16});
  EXPECT_EQ(lat.rank(), 3U);
}

TEST(SparsityLattice, ShapeMatchesConstructorInput) {
  proteus::SparsityLattice lat({4, 8, 16});
  auto s = lat.shape();
  ASSERT_EQ(s.size(), 3U);
  EXPECT_EQ(s[0], 4U);
  EXPECT_EQ(s[1], 8U);
  EXPECT_EQ(s[2], 16U);
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
  a[0].reset(1);
  b[0].reset(1);
  auto result = proteus::SparsityLattice::join(a, b);
  EXPECT_FALSE(result[0][1]);
}

TEST(SparsityLattice, JoinDropsSparseBitNotAgreedByBothOperands) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
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

TEST(SparsityLattice, MeetOfTwoDenseLatticesIsDense) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
  auto result = proteus::SparsityLattice::meet(a, b);
  EXPECT_TRUE(result[0].all());
  EXPECT_TRUE(result[1].all());
}

TEST(SparsityLattice, MeetKeepsSparseBitAgreedByBothOperands) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
  a[0].reset(1);
  b[0].reset(1);
  auto result = proteus::SparsityLattice::meet(a, b);
  EXPECT_FALSE(result[0][1]);
}

TEST(SparsityLattice, MeetPreservesSparseBitFromEitherOperand) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
  a[0].reset(2);
  auto result = proteus::SparsityLattice::meet(a, b);
  EXPECT_FALSE(result[0][2]);
  EXPECT_EQ(result, a);
}

TEST(SparsityLattice, MeetIsCommutative) {
  proteus::SparsityLattice a({4, 8});
  proteus::SparsityLattice b({4, 8});
  a[0].reset(0);
  b[1].reset(3);
  auto ab = proteus::SparsityLattice::meet(a, b);
  auto ba = proteus::SparsityLattice::meet(b, a);
  EXPECT_EQ(ab, ba);
}

TEST(SparsityLattice, MeetIsIdempotent) {
  proteus::SparsityLattice a({4, 8});
  a[0].reset(1);
  a[1].reset(5);
  auto result = proteus::SparsityLattice::meet(a, a);
  EXPECT_EQ(result, a);
}

TEST(SparsityLattice, FromAttrReturnsNulloptWhenKeyMissing) {
  mlir::MLIRContext ctx;
  auto emptyDict = mlir::DictionaryAttr::get(&ctx, {});
  EXPECT_FALSE(proteus::SparsityLattice::fromAttr(emptyDict).has_value());
}

TEST(SparsityLattice, FromAttrRoundTripFullyDense) {
  mlir::MLIRContext ctx;
  proteus::SparsityLattice original({4, 8});
  auto attr = proteus::SparsityLattice::toAttr(original, &ctx);
  auto str = mlir::StringAttr::get(&ctx, "proteus.lattice");
  auto dict = mlir::DictionaryAttr::get(&ctx, {{str, attr}});
  auto result = proteus::SparsityLattice::fromAttr(dict);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), original);
}

TEST(SparsityLattice, FromAttrRoundTripWithSparseBits) {
  mlir::MLIRContext ctx;
  proteus::SparsityLattice original({4, 8});
  original[0].reset(1);
  original[0].reset(3);
  original[1].reset(5);
  auto attr = proteus::SparsityLattice::toAttr(original, &ctx);
  auto str = mlir::StringAttr::get(&ctx, "proteus.lattice");
  auto dict = mlir::DictionaryAttr::get(&ctx, {{str, attr}});
  auto result = proteus::SparsityLattice::fromAttr(dict);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), original);
}

TEST(SparsityLattice, getDensityRangesTest) {
  llvm::BitVector bv(8, false);
  auto runs = proteus::SparsityLattice::getDensityRanges(bv);
  EXPECT_TRUE(runs.empty());
  bv.set(0);
  bv.set(1);
  runs = proteus::SparsityLattice::getDensityRanges(bv);
  ASSERT_EQ(runs.size(), 1U);
  EXPECT_EQ(runs[0], std::make_pair(int64_t{0}, int64_t{2}));
  bv.reset();
  bv.set(2);
  bv.set(3);
  bv.set(4);
  runs = proteus::SparsityLattice::getDensityRanges(bv);
  ASSERT_EQ(runs.size(), 1U);
  EXPECT_EQ(runs[0], std::make_pair(int64_t{2}, int64_t{3}));
  bv.reset();
  bv.set(0);
  bv.set(1);
  bv.set(4);
  bv.set(5);
  runs = proteus::SparsityLattice::getDensityRanges(bv);
  ASSERT_EQ(runs.size(), 2U);
  EXPECT_EQ(runs[0], std::make_pair(int64_t{0}, int64_t{2}));
  EXPECT_EQ(runs[1], std::make_pair(int64_t{4}, int64_t{2}));
}

TEST(SparsityLattice, FromAttrRoundTripAcrossWordBoundary) {
  mlir::MLIRContext ctx;
  llvm::SmallVector<uint64_t> shape{80U};
  proteus::SparsityLattice original(shape);
  original[0].reset(63);
  original[0].reset(64);
  auto attr = proteus::SparsityLattice::toAttr(original, &ctx);
  auto str = mlir::StringAttr::get(&ctx, "proteus.lattice");
  auto dict = mlir::DictionaryAttr::get(&ctx, {{str, attr}});
  auto result = proteus::SparsityLattice::fromAttr(dict);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), original);
}
