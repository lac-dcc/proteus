// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @broadcast_backward_last_dim
func.func @broadcast_backward_last_dim(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]}
    %input : tensor<4xf32>,
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 3 : i64, words = array<i64: 7>}]}
    %init : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: dimensions = [1] {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
  %0 = linalg.broadcast {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]} ins(%input : tensor<4xf32>) outs(%init : tensor<4x3xf32>) dimensions = [1]
  return %0 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @broadcast_backward_first_dim
func.func @broadcast_backward_first_dim(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]}
    %input : tensor<4xf32>,
    // CHECK: {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
    %init : tensor<3x4xf32>
) -> tensor<3x4xf32> {
  // CHECK: dimensions = [0] {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}]}
  %0 = linalg.broadcast {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%input : tensor<4xf32>) outs(%init : tensor<3x4xf32>) dimensions = [0]
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @broadcast_backward_first_last_dim
func.func @broadcast_backward_first_last_dim(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]}
    %input : tensor<4xf32>,
    // CHECK: {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}, {size = 2 : i64, words = array<i64: 3>}]}
    %init : tensor<3x4x2xf32>
) -> tensor<3x4x2xf32> {
  // CHECK: dimensions = [0, 2] {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 2 : i64, words = array<i64: 1>}]}
  %0 = linalg.broadcast {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 2 : i64, words = array<i64: 1>}]} ins(%input : tensor<4xf32>) outs(%init : tensor<3x4x2xf32>) dimensions = [0, 2]
  return %0 : tensor<3x4x2xf32>
}
