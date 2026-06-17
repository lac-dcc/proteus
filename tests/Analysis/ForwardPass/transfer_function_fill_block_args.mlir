// RUN: proteus-opt --spa-analysis="lattice-dump=true" %s | FileCheck %s

// CHECK-LABEL: func.func @fill_zero_test
func.func @fill_zero_test() -> tensor<3x4xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<3x4xf32>
  // CHECK: linalg.fill {proteus.lattice = [{size = 3 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
  %0 = linalg.fill ins(%cst : f32) outs(%empty : tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @fill_nonzero_test
func.func @fill_nonzero_test() -> tensor<3x4xf32> {
  %cst = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<3x4xf32>
  // CHECK: linalg.fill {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %0 = linalg.fill ins(%cst : f32) outs(%empty : tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
