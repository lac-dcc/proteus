// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s


// CHECK-LABEL: func.func @matmul_dense_accumulator_forces_density
func.func @matmul_dense_accumulator_forces_density(
    %lhs : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 14>}, {size = 3 : i64, words = array<i64: 7>}]},
    %rhs : tensor<3x4xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]},
    %init : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 8>}]}
) -> tensor<4x4xf32> {
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<4x3xf32>, tensor<3x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @matvec_dense_accumulator_forces_density
func.func @matvec_dense_accumulator_forces_density(
    %matrix : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 14>}, {size = 3 : i64, words = array<i64: 7>}]},
    %vec : tensor<3xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 1>}]}
) -> tensor<4xf32> {
  // CHECK: linalg.matvec {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
  %0 = linalg.matvec ins(%matrix, %vec : tensor<4x3xf32>, tensor<3xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @conv2d_dense_accumulator_forces_density
func.func @conv2d_dense_accumulator_forces_density(
    %input : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}]},
    %filter : tensor<2x2xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init : tensor<3x3xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 0>}]}
) -> tensor<3x3xf32> {
  // CHECK: linalg.conv_2d {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = linalg.conv_2d ins(%input, %filter : tensor<4x4xf32>, tensor<2x2xf32>) outs(%init : tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
