// RUN: proteus-opt --spa-analysis="lattice-dump=true" --spa-rewrite="target=scf" %s | FileCheck %s

// CHECK-LABEL: func.func @fully_sparse_batch
func.func @fully_sparse_batch(%arg0: tensor<1x2x4x4xf32>, %arg1: tensor<2x1x1xf32>) -> tensor<1x2x4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<1x2x4x4xf32>
  %0 = linalg.depthwise_conv_2d_nchw_chw {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%arg0, %arg1 : tensor<1x2x4x4xf32>, tensor<2x1x1xf32>) outs(%init : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %0 : tensor<1x2x4x4xf32>
}
// CHECK: %[[ZEROED:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<1x2x4x4xf32>
// CHECK-NOT: linalg.depthwise_conv_2d_nchw_chw
// CHECK-NOT: scf.for
// CHECK: return %[[ZEROED]]
