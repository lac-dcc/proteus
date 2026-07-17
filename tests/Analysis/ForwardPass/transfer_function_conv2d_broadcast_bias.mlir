// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @conv2d_broadcast_zero_bias_outs
func.func @conv2d_broadcast_zero_bias_outs(
    %input  : tensor<1x1x4x4xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}]},
    %filter : tensor<1x1x2x2xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init   : tensor<1x1x3x3xf32>
) -> tensor<1x1x3x3xf32> {
  %bias = arith.constant dense<0.000000e+00> : tensor<1xf32>
  // CHECK: linalg.broadcast ins(%{{.*}} : tensor<1xf32>) outs(%{{.*}} : tensor<1x1x3x3xf32>) dimensions = [0, 2, 3] {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 1 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}]}
  %outs = linalg.broadcast ins(%bias : tensor<1xf32>) outs(%init : tensor<1x1x3x3xf32>) dimensions = [0, 2, 3]
  // CHECK: linalg.conv_2d_nchw_fchw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 6>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = linalg.conv_2d_nchw_fchw ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) outs(%outs : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}
