// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @vecmat_backward_test
func.func @vecmat_backward_test(
    // The vector collapses entirely into each output element, so it has no
    // result dimension to invert from and stays fully dense.
    // CHECK: {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}]}
    %vec : tensor<3xf32>,
    // The result's N dim (col 3) is directly inherited from the matrix's N
    // dim, so it tightens; the K dim is untouched here since it is
    // LateralPass's job, not Backward's.
    // CHECK: {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
    %matrix : tensor<3x4xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
    %init : tensor<4xf32>
) -> tensor<4xf32> {
  // CHECK: linalg.vecmat {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]}
  %0 = linalg.vecmat {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]} ins(%vec, %matrix : tensor<3xf32>, tensor<3x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
