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

// CHECK-LABEL: func.func @array_1d_mixed
func.func @array_1d_mixed() -> tensor<3xf32> {
  // CHECK: arith.constant {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>}]}
  %c = arith.constant dense<[1.0, 0.0, 1.0]> : tensor<3xf32>
  return %c : tensor<3xf32>
}

// CHECK-LABEL: func.func @array_2d_mixed
func.func @array_2d_mixed() -> tensor<2x2xf32> {
  // CHECK: arith.constant {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 1>}]}
  %c = arith.constant dense<[[1.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @array_2d_diagonal
func.func @array_2d_diagonal() -> tensor<2x2xf32> {
  // CHECK: arith.constant {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
  %c = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @hex_all_zeros
func.func @hex_all_zeros() -> tensor<4xf32> {
  // CHECK: arith.constant {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
  %c = arith.constant dense<"0x00000000000000000000000000000000"> : tensor<4xf32>
  return %c : tensor<4xf32>
}

// CHECK-LABEL: func.func @hex_nonzero_i8
func.func @hex_nonzero_i8() -> tensor<4xi8> {
  // CHECK: arith.constant {proteus.lattice = [{size = 4 : i64, words = array<i64: 9>}]}
  %c = arith.constant dense<"0x01000001"> : tensor<4xi8>
  return %c : tensor<4xi8>
}

// CHECK-LABEL: func.func @hex_large_row_major
func.func @hex_large_row_major() -> tensor<4x8xi8> {
  // CHECK: arith.constant {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 8 : i64, words = array<i64: 9>}]}
  %c = arith.constant dense<"0x0100000000000000000000000000000000000001000000000000000000000000"> : tensor<4x8xi8>
  return %c : tensor<4x8xi8>
}

// CHECK-LABEL: func.func @hex_4d_row_major
func.func @hex_4d_row_major() -> tensor<2x2x2x4xi8> {
  // CHECK: arith.constant {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 6>}]}
  %c = arith.constant dense<"0x0001000000000000000000000000000000000000000000000000000000000100"> : tensor<2x2x2x4xi8>
  return %c : tensor<2x2x2x4xi8>
}

// CHECK-LABEL: func.func @array_3d_mixed
func.func @array_3d_mixed() -> tensor<2x3x2xf32> {
  // CHECK: arith.constant {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 5>}, {size = 2 : i64, words = array<i64: 1>}]}
  %c = arith.constant dense<[[[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
                              [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]> : tensor<2x3x2xf32>
  return %c : tensor<2x3x2xf32>
}
