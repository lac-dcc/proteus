// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_backward_test
func.func @matmul_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
    %lhs : tensor<4x4xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %rhs : tensor<4x4xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %b : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @self_matmul_backward
func.func @self_matmul_backward(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
    %A : tensor<4x4xf32>,
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%A, %A : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %b : tensor<4x4xf32>
}
