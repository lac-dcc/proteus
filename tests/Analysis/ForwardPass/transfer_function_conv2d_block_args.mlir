// RUN: proteus-opt --spa-analysis="lattice-dump=true" %s | FileCheck %s

// CHECK-LABEL: func.func @conv2d_row_sparsity_test
func.func @conv2d_row_sparsity_test(
    %input : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}]},
    %filter : tensor<2x2xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init : tensor<3x3xf32>
) -> tensor<3x3xf32> {
  // CHECK: linalg.conv_2d {proteus.lattice = [{size = 3 : i64, words = array<i64: 6>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = linalg.conv_2d ins(%input, %filter : tensor<4x4xf32>, tensor<2x2xf32>) outs(%init : tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: func.func @conv2d_col_sparsity_test
func.func @conv2d_col_sparsity_test(
    %input : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 12>}]},
    %filter : tensor<2x2xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init : tensor<3x3xf32>
) -> tensor<3x3xf32> {
  // CHECK: linalg.conv_2d {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 6>}]}
  %0 = linalg.conv_2d ins(%input, %filter : tensor<4x4xf32>, tensor<2x2xf32>) outs(%init : tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
