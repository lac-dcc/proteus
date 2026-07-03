// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @batch_matmul_backward_test
func.func @batch_matmul_backward_test(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
    %lhs : tensor<2x4x4xf32>,
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %rhs : tensor<2x4x4xf32>,
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
    %init : tensor<2x4x4xf32>
) -> tensor<2x4x4xf32> {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%lhs, %rhs : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %b : tensor<2x4x4xf32>
}

// CHECK-LABEL: func.func @batch_matmul_backward_batch_dim_ambiguous
func.func @batch_matmul_backward_batch_dim_ambiguous(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
    %lhs : tensor<2x4x4xf32>,
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
    %rhs : tensor<2x4x4xf32>,
    %init : tensor<2x4x4xf32>
) -> tensor<2x4x4xf32> {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]} ins(%lhs, %rhs : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %b : tensor<2x4x4xf32>
}

// CHECK-LABEL: func.func @self_batch_matmul_backward
func.func @self_batch_matmul_backward(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
    %A : tensor<2x4x4xf32>,
    %init : tensor<2x4x4xf32>
) -> tensor<2x4x4xf32> {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%A, %A : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %b : tensor<2x4x4xf32>
}
