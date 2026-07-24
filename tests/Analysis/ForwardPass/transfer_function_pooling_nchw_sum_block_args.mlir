// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @pooling_nchw_sum_channel_sparsity
func.func @pooling_nchw_sum_channel_sparsity(
    %input  : tensor<1x2x3x3xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 2>}, {size = 3 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 7>}]},
    %kernel : tensor<2x2xf32>,
    %init   : tensor<1x2x2x2xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 2 : i64, words = array<i64: 0>}, {size = 2 : i64, words = array<i64: 0>}, {size = 2 : i64, words = array<i64: 0>}]}
) -> tensor<1x2x2x2xf32> {
  // CHECK: linalg.pooling_nchw_sum {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 2>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
  %0 = linalg.pooling_nchw_sum ins(%input, %kernel : tensor<1x2x3x3xf32>, tensor<2x2xf32>) outs(%init : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %0 : tensor<1x2x2x2xf32>
}

// CHECK-LABEL: func.func @pooling_nchw_sum_h_sparsity
func.func @pooling_nchw_sum_h_sparsity(
    %input  : tensor<1x1x4x4xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}]},
    %kernel : tensor<2x2xf32>,
    %init   : tensor<1x1x3x3xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 1 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}]}
) -> tensor<1x1x3x3xf32> {
  // CHECK: linalg.pooling_nchw_sum {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 6>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = linalg.pooling_nchw_sum ins(%input, %kernel : tensor<1x1x4x4xf32>, tensor<2x2xf32>) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

// CHECK-LABEL: func.func @pooling_nchw_sum_stride2
func.func @pooling_nchw_sum_stride2(
    %input  : tensor<1x1x6x4xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 6 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}]},
    %kernel : tensor<2x2xf32>,
    %init   : tensor<1x1x3x3xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 1 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}]}
) -> tensor<1x1x3x3xf32> {
  // CHECK: linalg.pooling_nchw_sum {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 2>}, {size = 3 : i64, words = array<i64: 7>}], strides = dense<[2, 1]> : vector<2xi64>}
  %0 = linalg.pooling_nchw_sum {strides = dense<[2, 1]> : vector<2xi64>} ins(%input, %kernel : tensor<1x1x6x4xf32>, tensor<2x2xf32>) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

// CHECK-LABEL: func.func @pooling_nchw_sum_dilation2
func.func @pooling_nchw_sum_dilation2(
    %input  : tensor<1x1x6x4xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 6 : i64, words = array<i64: 42>}, {size = 4 : i64, words = array<i64: 15>}]},
    %kernel : tensor<2x2xf32>,
    %init   : tensor<1x1x4x3xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 1 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}]}
) -> tensor<1x1x4x3xf32> {
  // CHECK: linalg.pooling_nchw_sum {dilations = dense<[2, 1]> : vector<2xi64>, proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = linalg.pooling_nchw_sum {dilations = dense<[2, 1]> : vector<2xi64>} ins(%input, %kernel : tensor<1x1x6x4xf32>, tensor<2x2xf32>) outs(%init : tensor<1x1x4x3xf32>) -> tensor<1x1x4x3xf32>
  return %0 : tensor<1x1x4x3xf32>
}
