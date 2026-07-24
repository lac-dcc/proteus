// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @matvec_backward_test
func.func @matvec_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 7>}]}
    %matrix : tensor<4x3xf32>,
    // CHECK: {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}]}
    %vec : tensor<3xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  // CHECK: linalg.matvec {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]}
  %0 = linalg.matvec {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]} ins(%matrix, %vec : tensor<4x3xf32>, tensor<3xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
