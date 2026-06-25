// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @div_test
func.func @div_test(
    %lhs : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %rhs : tensor<4x3xf32>,
    %init : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: linalg.div {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]}
  %0 = linalg.div ins(%lhs, %rhs : tensor<4x3xf32>, tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}
