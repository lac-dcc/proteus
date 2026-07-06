// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @add_backward_test
func.func @add_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
    %lhs : tensor<4x3xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
    %rhs : tensor<4x3xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 3 : i64, words = array<i64: 7>}]}
    %init : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: linalg.add {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
  %0 = linalg.add {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]} ins(%lhs, %rhs : tensor<4x3xf32>, tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @self_add_backward
func.func @self_add_backward(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
    %A : tensor<4x3xf32>,
    %init : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: linalg.add {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
  %0 = linalg.add {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]} ins(%A, %A : tensor<4x3xf32>, tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @add_backward_fixpoint_chain
func.func @add_backward_fixpoint_chain(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
    %a : tensor<4x3xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
    %r0 : tensor<4x3xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
    %r1 : tensor<4x3xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
    %r2 : tensor<4x3xf32>,
    %init : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: linalg.add {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
  %b1 = linalg.add ins(%a, %r0 : tensor<4x3xf32>, tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  // CHECK: linalg.add {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
  %b2 = linalg.add ins(%b1, %r1 : tensor<4x3xf32>, tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  // CHECK: linalg.add {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
  %b3 = linalg.add {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]} ins(%b2, %r2 : tensor<4x3xf32>, tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  return %b3 : tensor<4x3xf32>
}
