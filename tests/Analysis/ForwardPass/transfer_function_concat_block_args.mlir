// RUN: proteus-opt --spa-analysis="lattice-dump=true" %s | FileCheck %s

// CHECK-LABEL: func.func @concat_1d_along_dim0
func.func @concat_1d_along_dim0(
    %a : tensor<3xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>}]},
    %b : tensor<3xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 2>}]}
) -> tensor<6xf32> {
  // CHECK: tensor.concat dim(0){{.*}}{proteus.lattice = [{size = 6 : i64, words = array<i64: 21>}]}
  %0 = tensor.concat dim(0) %a, %b : (tensor<3xf32>, tensor<3xf32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func.func @concat_2d_along_dim1
func.func @concat_2d_along_dim1(
    %a : tensor<2x3xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 5>}]},
    %b : tensor<2x2xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 2>}, {size = 2 : i64, words = array<i64: 1>}]}
) -> tensor<2x5xf32> {
  // CHECK: tensor.concat dim(1){{.*}}{proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 5 : i64, words = array<i64: 13>}]}
  %0 = tensor.concat dim(1) %a, %b : (tensor<2x3xf32>, tensor<2x2xf32>) -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}

// CHECK-LABEL: func.func @concat_2d_non_concat_dim_or_semantics
func.func @concat_2d_non_concat_dim_or_semantics(
    %a : tensor<2x3xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}]},
    %b : tensor<2x2xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 2>}, {size = 2 : i64, words = array<i64: 3>}]}
) -> tensor<2x5xf32> {
  // CHECK: tensor.concat dim(1){{.*}}{proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 5 : i64, words = array<i64: 31>}]}
  %0 = tensor.concat dim(1) %a, %b : (tensor<2x3xf32>, tensor<2x2xf32>) -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}

// CHECK-LABEL: func.func @concat_3d_along_dim1
func.func @concat_3d_along_dim1(
    %a : tensor<2x3x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 10>}]},
    %b : tensor<2x2x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 2>}, {size = 4 : i64, words = array<i64: 5>}]}
) -> tensor<2x5x4xf32> {
  // CHECK: tensor.concat dim(1){{.*}}{proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 5 : i64, words = array<i64: 21>}, {size = 4 : i64, words = array<i64: 15>}]}
  %0 = tensor.concat dim(1) %a, %b : (tensor<2x3x4xf32>, tensor<2x2x4xf32>) -> tensor<2x5x4xf32>
  return %0 : tensor<2x5x4xf32>
}
