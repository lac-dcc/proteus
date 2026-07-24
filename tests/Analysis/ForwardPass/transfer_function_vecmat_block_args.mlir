// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @vecmat_test
func.func @vecmat_test(
    %vec : tensor<3xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>}]},
    %matrix : tensor<3x4xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>},
                                                   {size = 4 : i64, words = array<i64: 10>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  // CHECK: linalg.vecmat {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}]}
  %0 = linalg.vecmat ins(%vec, %matrix : tensor<3xf32>, tensor<3x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
