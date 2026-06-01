#include "Analysis/SparsityLattice.h"

#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

TEST(SparsityLattice, ConstructionSetsAllSparse) {
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
