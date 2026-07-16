// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s


// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_backward_stride
func.func @conv2d_nhwc_hwcf_backward_stride(
    // CHECK: {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 6 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 12>}, {size = 1 : i64, words = array<i64: 1>}]}
    %input : tensor<1x6x4x1xf32>,
    %filter : tensor<2x2x1x1xf32>,
    %init : tensor<1x3x3x1xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}, {size = 1 : i64, words = array<i64: 0>}]}
) -> tensor<1x3x3x1xf32> {
  // CHECK: linalg.conv_2d_nhwc_hwcf {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 6>}, {size = 1 : i64, words = array<i64: 1>}], strides = dense<[2, 1]> : vector<2xi64>}
  %0 = linalg.conv_2d_nhwc_hwcf {strides = dense<[2, 1]> : vector<2xi64>, proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 6>}, {size = 1 : i64, words = array<i64: 1>}]} ins(%input, %filter : tensor<1x6x4x1xf32>, tensor<2x2x1x1xf32>) outs(%init : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  return %0 : tensor<1x3x3x1xf32>
}
