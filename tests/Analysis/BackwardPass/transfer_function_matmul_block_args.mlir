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

// CHECK-LABEL: func.func @matmul_backward_lhs_chain
func.func @matmul_backward_lhs_chain(
    // Two hops away from the forced-sparse result: row 3 still reaches all the way back.
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
    %a : tensor<4x4xf32>,
    %r0 : tensor<4x4xf32>,
    %r1 : tensor<4x4xf32>,
    %r2 : tensor<4x4xf32>,
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b1 = linalg.matmul ins(%a, %r0 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b2 = linalg.matmul ins(%b1, %r1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b3 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]} ins(%b2, %r2 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %b3 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @matmul_backward_rhs_chain
func.func @matmul_backward_rhs_chain(
    %l0 : tensor<4x4xf32>,
    %l1 : tensor<4x4xf32>,
    %l2 : tensor<4x4xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xf32>,
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %c1 = linalg.matmul ins(%l0, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %c2 = linalg.matmul ins(%l1, %c1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %c3 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%l2, %c2 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c3 : tensor<4x4xf32>
}
