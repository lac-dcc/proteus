// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @collapse_shape_all_dense
func.func @collapse_shape_all_dense(
    %src : tensor<2x3xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 7>}]}
) -> tensor<6xf32> {
  // CHECK: tensor.collapse_shape {{.*}} {proteus.lattice = [{size = 6 : i64, words = array<i64: 63>}]}
  %0 = tensor.collapse_shape %src [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func.func @collapse_shape_all_sparse
func.func @collapse_shape_all_sparse(
    %src : tensor<2x3xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}]}
) -> tensor<6xf32> {
  // CHECK: tensor.collapse_shape {{.*}} {proteus.lattice = [{size = 6 : i64, words = array<i64: 0>}]}
  %0 = tensor.collapse_shape %src [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func.func @collapse_shape_sparse_row
func.func @collapse_shape_sparse_row(
    %src : tensor<2x3xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}]}
) -> tensor<6xf32> {
  // CHECK: tensor.collapse_shape {{.*}} {proteus.lattice = [{size = 6 : i64, words = array<i64: 7>}]}
  %0 = tensor.collapse_shape %src [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func.func @collapse_shape_sparse_col
func.func @collapse_shape_sparse_col(
    %src : tensor<2x3xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 3 : i64, words = array<i64: 5>}]}
) -> tensor<6xf32> {
  // CHECK: tensor.collapse_shape {{.*}} {proteus.lattice = [{size = 6 : i64, words = array<i64: 45>}]}
  %0 = tensor.collapse_shape %src [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func.func @collapse_shape_passthrough_and_collapse
func.func @collapse_shape_passthrough_and_collapse(
    %src : tensor<2x2x3xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}]}
) -> tensor<2x6xf32> {
  // CHECK: tensor.collapse_shape {{.*}} {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 6 : i64, words = array<i64: 7>}]}
  %0 = tensor.collapse_shape %src [[0], [1, 2]] : tensor<2x2x3xf32> into tensor<2x6xf32>
  return %0 : tensor<2x6xf32>
}

// CHECK-LABEL: func.func @collapse_shape_mixed_sparsity
func.func @collapse_shape_mixed_sparsity(
    %src : tensor<2x4xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 9>}]}
) -> tensor<8xf32> {
  // CHECK: tensor.collapse_shape {{.*}} {proteus.lattice = [{size = 8 : i64, words = array<i64: 9>}]}
  %0 = tensor.collapse_shape %src [[0, 1]] : tensor<2x4xf32> into tensor<8xf32>
  return %0 : tensor<8xf32>
}
