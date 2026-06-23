// RUN: proteus-opt --spa-analysis="lattice-dump=true" %s | FileCheck %s

// CHECK-LABEL: func.func @expand_shape_general_split
func.func @expand_shape_general_split(
    %src : tensor<1x4xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 5>}]}
) -> tensor<1x2x2xf32> {
  // CHECK: tensor.expand_shape {{.*}} {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 1>}]}
  %0 = tensor.expand_shape %src [[0], [1, 2]] output_shape [1, 2, 2] : tensor<1x4xf32> into tensor<1x2x2xf32>
  return %0 : tensor<1x2x2xf32>
}

// CHECK-LABEL: func.func @expand_shape_all_sparse
func.func @expand_shape_all_sparse(
    %src : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<2x2xf32> {
  // CHECK: tensor.expand_shape {{.*}} {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 2 : i64, words = array<i64: 0>}]}
  %0 = tensor.expand_shape %src [[0, 1]] output_shape [2, 2] : tensor<4xf32> into tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @expand_shape_all_dense
func.func @expand_shape_all_dense(
    %src : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
) -> tensor<2x2xf32> {
  // CHECK: tensor.expand_shape {{.*}} {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
  %0 = tensor.expand_shape %src [[0, 1]] output_shape [2, 2] : tensor<4xf32> into tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @expand_shape_partial_first_row_dense
func.func @expand_shape_partial_first_row_dense(
    %src : tensor<6xf32> {proteus.lattice = [{size = 6 : i64, words = array<i64: 7>}]}
) -> tensor<2x3xf32> {
  // CHECK: tensor.expand_shape {{.*}} {proteus.lattice = [{size = 2 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = tensor.expand_shape %src [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @expand_shape_passthrough_and_split
func.func @expand_shape_passthrough_and_split(
    %src : tensor<2x6xf32> {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 6 : i64, words = array<i64: 7>}]}
) -> tensor<2x2x3xf32> {
  // CHECK: tensor.expand_shape {{.*}} {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = tensor.expand_shape %src [[0], [1, 2]] output_shape [2, 2, 3] : tensor<2x6xf32> into tensor<2x2x3xf32>
  return %0 : tensor<2x2x3xf32>
}
