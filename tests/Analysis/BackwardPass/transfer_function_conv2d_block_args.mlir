// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @conv2d_backward_row_sparsity
func.func @conv2d_backward_row_sparsity(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 12>}]}
    %input : tensor<4x4xf32>,
    %filter : tensor<2x2xf32>,
    %init : tensor<3x3xf32>
) -> tensor<3x3xf32> {
  // CHECK: linalg.conv_2d {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 6>}]}
  %0 = linalg.conv_2d {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 6>}]} ins(%input, %filter : tensor<4x4xf32>, tensor<2x2xf32>) outs(%init : tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
