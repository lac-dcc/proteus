// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_filter_sparsity
func.func @conv2d_nhwc_hwcf_filter_sparsity(
    %input  : tensor<1x3x3x2xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 7>}, {size = 2 : i64, words = array<i64: 3>}]},
    %filter : tensor<2x2x2x3xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 6>}]},
    %init   : tensor<1x2x2x3xf32>
) -> tensor<1x2x2x3xf32> {
  // CHECK: linalg.conv_2d_nhwc_hwcf {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 6>}]}
  %0 = linalg.conv_2d_nhwc_hwcf ins(%input, %filter : tensor<1x3x3x2xf32>, tensor<2x2x2x3xf32>) outs(%init : tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
  return %0 : tensor<1x2x2x3xf32>
}

// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_h_sparsity
func.func @conv2d_nhwc_hwcf_h_sparsity(
    %input  : tensor<1x4x4x1xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}, {size = 1 : i64, words = array<i64: 1>}]},
    %filter : tensor<2x2x1x1xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}, {size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}]},
    %init   : tensor<1x3x3x1xf32>
) -> tensor<1x3x3x1xf32> {
  // CHECK: linalg.conv_2d_nhwc_hwcf {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 6>}, {size = 3 : i64, words = array<i64: 7>}, {size = 1 : i64, words = array<i64: 1>}]}
  %0 = linalg.conv_2d_nhwc_hwcf ins(%input, %filter : tensor<1x4x4x1xf32>, tensor<2x2x1x1xf32>) outs(%init : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  return %0 : tensor<1x3x3x1xf32>
}

// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_stride2
func.func @conv2d_nhwc_hwcf_stride2(
    %input  : tensor<1x6x4x1xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 6 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}, {size = 1 : i64, words = array<i64: 1>}]},
    %filter : tensor<2x2x1x1xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}, {size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}]},
    %init   : tensor<1x3x3x1xf32>
) -> tensor<1x3x3x1xf32> {
  // CHECK: linalg.conv_2d_nhwc_hwcf {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 2>}, {size = 3 : i64, words = array<i64: 7>}, {size = 1 : i64, words = array<i64: 1>}], strides = dense<[2, 1]> : vector<2xi64>}
  %0 = linalg.conv_2d_nhwc_hwcf {strides = dense<[2, 1]> : vector<2xi64>} ins(%input, %filter : tensor<1x6x4x1xf32>, tensor<2x2x1x1xf32>) outs(%init : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  return %0 : tensor<1x3x3x1xf32>
}
