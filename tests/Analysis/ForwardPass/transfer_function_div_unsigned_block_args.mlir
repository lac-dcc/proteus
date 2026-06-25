// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @div_unsigned_test
func.func @div_unsigned_test(
    %lhs : tensor<4x3xi32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %rhs : tensor<4x3xi32>,
    %init : tensor<4x3xi32>
) -> tensor<4x3xi32> {
  // CHECK: linalg.div_unsigned {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]}
  %0 = linalg.div_unsigned ins(%lhs, %rhs : tensor<4x3xi32>, tensor<4x3xi32>) outs(%init : tensor<4x3xi32>) -> tensor<4x3xi32>
  return %0 : tensor<4x3xi32>
}
