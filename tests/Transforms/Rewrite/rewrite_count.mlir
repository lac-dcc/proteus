// RUN: proteus-opt --spa-rewrite="count-rewrites=true" %s -o /dev/null 2>&1 | FileCheck %s

// Two linalg.matmul ops (one split into sub-blocks, one replaced by a zero
// fill) and one linalg.conv_2d_nchw_fchw op (replaced by a zero fill) should
// be tallied per rewrite rule: matmul=2, conv2d=1.
func.func @counts(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>,
                  %input: tensor<1x1x4x4xf32>, %filter: tensor<1x1x1x1xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<1x1x4x4xf32>) {
  %init0 = arith.constant dense<0.0> : tensor<4x4xf32>
  %m0 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %init1 = arith.constant dense<0.0> : tensor<4x4xf32>
  %m1 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %init2 = arith.constant dense<0.0> : tensor<1x1x4x4xf32>
  %c0 = linalg.conv_2d_nchw_fchw {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32>) outs(%init2 : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %m0, %m1, %c0 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<1x1x4x4xf32>
}

// CHECK: spa-rewrite counts for @counts: linalg.matmul=2, linalg.conv_2d_nchw_fchw=1
