// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=lateral" %s | FileCheck %s

// CHECK-LABEL: func.func @batch_matmul_test
func.func @batch_matmul_test(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 7>}]}
    %lhs : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},

    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
    %rhs : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]},

    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 15>}]}
    %3   : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},

    %init : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<2x4x4xf32> {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 5>}]}
  %0 = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  %1 = linalg.batch_matmul ins(%0, %3 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %0 : tensor<2x4x4xf32>
}
