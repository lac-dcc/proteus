// RUN: proteus-opt --spa-rewrite="time-rewrite=true" %s -o /dev/null 2>&1 | FileCheck %s

func.func @single_dense_block(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<4x4xf32>
  %0 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK: Execution time report
// CHECK: Total Execution Time
// CHECK: Rewrite
// CHECK: Total
