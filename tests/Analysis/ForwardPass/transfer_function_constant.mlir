// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @splat_zero_float
func.func @splat_zero_float() -> tensor<3x4xf32> {
  // CHECK: arith.constant {proteus.lattice = [{size = 3 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
  %c = arith.constant dense<0.0> : tensor<3x4xf32>
  return %c : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @splat_nonzero_float
func.func @splat_nonzero_float() -> tensor<3x4xf32> {
  // CHECK: arith.constant {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
  %c = arith.constant dense<1.0> : tensor<3x4xf32>
  return %c : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @splat_zero_int
func.func @splat_zero_int() -> tensor<4x2xi32> {
  // CHECK: arith.constant {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 2 : i64, words = array<i64: 0>}]}
  %c = arith.constant dense<0> : tensor<4x2xi32>
  return %c : tensor<4x2xi32>
}

// CHECK-LABEL: func.func @array_2d_diagonal
func.func @array_2d_diagonal() -> tensor<2x2xf32> {
  // CHECK: arith.constant {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
  %c = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}
