// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @extract_slice_1d_sparse_region
func.func @extract_slice_1d_sparse_region(
    %src : tensor<8xf32> {proteus.lattice = [{size = 8 : i64, words = array<i64: 15>}]}
) -> tensor<4xf32> {
  // CHECK: tensor.extract_slice {{.*}} {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
  %0 = tensor.extract_slice %src[4][4][1] : tensor<8xf32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @extract_slice_1d_dense_region
func.func @extract_slice_1d_dense_region(
    %src : tensor<8xf32> {proteus.lattice = [{size = 8 : i64, words = array<i64: 15>}]}
) -> tensor<4xf32> {
  // CHECK: tensor.extract_slice {{.*}} {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
  %0 = tensor.extract_slice %src[0][4][1] : tensor<8xf32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @extract_slice_1d_stride
func.func @extract_slice_1d_stride(
    %src : tensor<8xf32> {proteus.lattice = [{size = 8 : i64, words = array<i64: 85>}]}
) -> tensor<4xf32> {
  // CHECK: tensor.extract_slice {{.*}} {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
  %0 = tensor.extract_slice %src[1][4][2] : tensor<8xf32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @extract_slice_1d_stride_dense
func.func @extract_slice_1d_stride_dense(
    %src : tensor<8xf32> {proteus.lattice = [{size = 8 : i64, words = array<i64: 85>}]}
) -> tensor<4xf32> {
  // CHECK: tensor.extract_slice {{.*}} {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
  %0 = tensor.extract_slice %src[0][4][2] : tensor<8xf32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @extract_slice_2d_dense_rows
func.func @extract_slice_2d_dense_rows(
    %src : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]}
) -> tensor<2x4xf32> {
  // CHECK: tensor.extract_slice {{.*}} {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]}
  %0 = tensor.extract_slice %src[0, 0][2, 4][1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @extract_slice_2d_sparse_rows
func.func @extract_slice_2d_sparse_rows(
    %src : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]}
) -> tensor<2x4xf32> {
  // CHECK: tensor.extract_slice {{.*}} {proteus.lattice = [{size = 2 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 15>}]}
  %0 = tensor.extract_slice %src[2, 0][2, 4][1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @extract_slice_rank_reduction
func.func @extract_slice_rank_reduction(
    %src : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 9>}, {size = 4 : i64, words = array<i64: 6>}]}
) -> tensor<4xf32> {
  // CHECK: tensor.extract_slice {{.*}} {proteus.lattice = [{size = 4 : i64, words = array<i64: 6>}]}
  %0 = tensor.extract_slice %src[1, 0][1, 4][1, 1] : tensor<4x4xf32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}
