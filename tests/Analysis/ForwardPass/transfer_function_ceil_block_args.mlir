// RUN: proteus-opt --spa-analysis="lattice-dump=true" %s | FileCheck %s

// CHECK-LABEL: func.func @ceil_test
func.func @ceil_test(
    %input : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %init : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: linalg.ceil {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]}
  %0 = linalg.ceil ins(%input : tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}
