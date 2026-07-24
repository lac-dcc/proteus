// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @pooling_nchw_sum_backward_stride
func.func @pooling_nchw_sum_backward_stride(
    // CHECK: {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 6 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 3>}]}
    %input : tensor<1x1x6x4xf32>,
    %kernel : tensor<2x2xf32>,
    %init : tensor<1x1x3x3xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 1 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}]}
) -> tensor<1x1x3x3xf32> {
  // CHECK: linalg.pooling_nchw_sum {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 3>}], strides = dense<[2, 1]> : vector<2xi64>}
  %0 = linalg.pooling_nchw_sum {strides = dense<[2, 1]> : vector<2xi64>, proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 3>}]} ins(%input, %kernel : tensor<1x1x6x4xf32>, tensor<2x2xf32>) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}
