// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_test
func.func @matmul_test(
    %lhs : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>},
                                               {size = 3 : i64, words = array<i64: 7>}]},
    %rhs : tensor<3x4xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>},
                                               {size = 4 : i64, words = array<i64: 5>}]},
    %init : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>},
                                                {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4x4xf32> {
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 4>}]}
  %0 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 11>}, {size = 4 : i64, words = array<i64: 14>}]} ins(%lhs, %rhs : tensor<4x3xf32>, tensor<3x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
