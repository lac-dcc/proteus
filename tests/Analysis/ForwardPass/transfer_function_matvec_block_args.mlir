// RUN: proteus-opt --spa-analysis="lattice-dump=true" %s | FileCheck %s

// CHECK-LABEL: func.func @matvec_test
func.func @matvec_test(
    %matrix : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>},
                                                   {size = 3 : i64, words = array<i64: 5>}]},
    %vec : tensor<3xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>}]},
    %init : tensor<4xf32>
) -> tensor<4xf32> {
  // CHECK: linalg.matvec {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}]}
  %0 = linalg.matvec ins(%matrix, %vec : tensor<4x3xf32>, tensor<3xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
