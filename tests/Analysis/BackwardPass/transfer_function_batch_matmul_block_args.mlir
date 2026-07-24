// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @batch_matmul_backward_test
func.func @batch_matmul_backward_test(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
    %lhs : tensor<2x4x4xf32>,
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %rhs : tensor<2x4x4xf32>,
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
    %init : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
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
    %init : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<2x4x4xf32> {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]} ins(%lhs, %rhs : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %b : tensor<2x4x4xf32>
}

// CHECK-LABEL: func.func @self_batch_matmul_backward
func.func @self_batch_matmul_backward(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
    %A : tensor<2x4x4xf32>,
    %init : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<2x4x4xf32> {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%A, %A : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %b : tensor<2x4x4xf32>
}

// CHECK-LABEL: func.func @batch_matmul_backward_lhs_chain
func.func @batch_matmul_backward_lhs_chain(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
    %a : tensor<2x4x4xf32>,
    %r0 : tensor<2x4x4xf32>,
    %r1 : tensor<2x4x4xf32>,
    %r2 : tensor<2x4x4xf32>,
    %init : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<2x4x4xf32> {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b1 = linalg.batch_matmul ins(%a, %r0 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b2 = linalg.batch_matmul ins(%b1, %r1 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b3 = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]} ins(%b2, %r2 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %b3 : tensor<2x4x4xf32>
}

// CHECK-LABEL: func.func @batch_matmul_backward_rhs_chain
func.func @batch_matmul_backward_rhs_chain(
    %l0 : tensor<2x4x4xf32>,
    %l1 : tensor<2x4x4xf32>,
    %l2 : tensor<2x4x4xf32>,
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<2x4x4xf32>,
    %init : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<2x4x4xf32> {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %c1 = linalg.batch_matmul ins(%l0, %a : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %c2 = linalg.batch_matmul ins(%l1, %c1 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %c3 = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%l2, %c2 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %c3 : tensor<2x4x4xf32>
}

// CHECK-LABEL: func.func @batch_matmul_backward_batch_dim_stays_ambiguous
func.func @batch_matmul_backward_batch_dim_stays_ambiguous(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
    %a : tensor<2x4x4xf32>,
    %r0 : tensor<2x4x4xf32>,
    %r1 : tensor<2x4x4xf32>,
    %r2 : tensor<2x4x4xf32>,
    %init : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<2x4x4xf32> {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b1 = linalg.batch_matmul ins(%a, %r0 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b2 = linalg.batch_matmul ins(%b1, %r1 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b3 = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]} ins(%b2, %r2 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %b3 : tensor<2x4x4xf32>
}

// CHECK-LABEL: func.func @batch_matmul_backward_users_agree
func.func @batch_matmul_backward_users_agree(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
    %a : tensor<2x4x4xf32>,
    %r0 : tensor<2x4x4xf32>,
    %r1 : tensor<2x4x4xf32>,
    %init : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> (tensor<2x4x4xf32>, tensor<2x4x4xf32>) {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b1 = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]} ins(%a, %r0 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b2 = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]} ins(%a, %r1 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %b1, %b2 : tensor<2x4x4xf32>, tensor<2x4x4xf32>
}

// CHECK-LABEL: func.func @batch_matmul_backward_users_disagree
func.func @batch_matmul_backward_users_disagree(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
    %a : tensor<2x4x4xf32>,
    %r0 : tensor<2x4x4xf32>,
    %r1 : tensor<2x4x4xf32>,
    %init : tensor<2x4x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> (tensor<2x4x4xf32>, tensor<2x4x4xf32>) {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b1 = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]} ins(%a, %r0 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %b2 = linalg.batch_matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]} ins(%a, %r1 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %b1, %b2 : tensor<2x4x4xf32>, tensor<2x4x4xf32>
}
