// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @conv2d_nchw_fchw_filter_sparsity
func.func @conv2d_nchw_fchw_filter_sparsity(
    %input  : tensor<1x2x3x3xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 7>}]},
    %filter : tensor<3x2x2x2xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 6>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init   : tensor<1x3x2x2xf32>
) -> tensor<1x3x2x2xf32> {
  // CHECK: linalg.conv_2d_nchw_fchw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 6>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
  %0 = linalg.conv_2d_nchw_fchw ins(%input, %filter : tensor<1x2x3x3xf32>, tensor<3x2x2x2xf32>) outs(%init : tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %0 : tensor<1x3x2x2xf32>
}

// CHECK-LABEL: func.func @conv2d_nchw_fchw_h_sparsity
func.func @conv2d_nchw_fchw_h_sparsity(
    %input  : tensor<1x1x4x4xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}]},
    %filter : tensor<1x1x2x2xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init   : tensor<1x1x3x3xf32>
) -> tensor<1x1x3x3xf32> {
  // CHECK: linalg.conv_2d_nchw_fchw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 6>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = linalg.conv_2d_nchw_fchw ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

// CHECK-LABEL: func.func @conv2d_nchw_fchw_stride2
func.func @conv2d_nchw_fchw_stride2(
    %input  : tensor<1x1x6x4xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 6 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}]},
    %filter : tensor<1x1x2x2xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init   : tensor<1x1x3x3xf32>
) -> tensor<1x1x3x3xf32> {
  // CHECK: linalg.conv_2d_nchw_fchw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 2>}, {size = 3 : i64, words = array<i64: 7>}], strides = dense<[2, 1]> : vector<2xi64>}
  %0 = linalg.conv_2d_nchw_fchw {strides = dense<[2, 1]> : vector<2xi64>} ins(%input, %filter : tensor<1x1x6x4xf32>, tensor<1x1x2x2xf32>) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}
