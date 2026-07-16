// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @conv2d_nchw_fchw_backward
func.func @conv2d_nchw_fchw_backward(
    // CHECK: {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]}
    %input : tensor<1x1x4x4xf32>,
    %filter : tensor<1x1x2x2xf32>,
    %init : tensor<1x1x3x3xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 1 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}]}
) -> tensor<1x1x3x3xf32> {
  // CHECK: linalg.conv_2d_nchw_fchw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = linalg.conv_2d_nchw_fchw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 7>}]} ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

// CHECK-LABEL: func.func @conv2d_nchw_fchw_backward_stride
func.func @conv2d_nchw_fchw_backward_stride(
    // CHECK: {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 6 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 12>}]}
    %input : tensor<1x1x6x4xf32>,
    %filter : tensor<1x1x2x2xf32>,
    %init : tensor<1x1x3x3xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 1 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}]}
) -> tensor<1x1x3x3xf32> {
  // CHECK: linalg.conv_2d_nchw_fchw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 6>}], strides = dense<[2, 1]> : vector<2xi64>}
  %0 = linalg.conv_2d_nchw_fchw 
  {strides = dense<[2, 1]> : vector<2xi64>, proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 6>}]} ins(%input, %filter : tensor<1x1x6x4xf32>, tensor<1x1x2x2xf32>) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}
