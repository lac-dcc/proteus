// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @broadcast_first_dim
func.func @broadcast_first_dim(
    %input : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}]},
    %init  : tensor<3x4xf32>
) -> tensor<3x4xf32> {
  // CHECK: dimensions = [0] {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 5>}]}
  %0 = linalg.broadcast ins(%input : tensor<4xf32>) outs(%init : tensor<3x4xf32>) dimensions = [0]
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @broadcast_last_dim
func.func @broadcast_last_dim(
    %input : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}]},
    %init  : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: dimensions = [1] {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = linalg.broadcast ins(%input : tensor<4xf32>) outs(%init : tensor<4x3xf32>) dimensions = [1]
  return %0 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @broadcast_first_last_dim
func.func @broadcast_first_last_dim(
    %input : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}]},
    %init  : tensor<3x4x2xf32>
) -> tensor<3x4x2xf32> {
  // CHECK: dimensions = [0, 2] {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 5>}, {size = 2 : i64, words = array<i64: 3>}]}
  %0 = linalg.broadcast ins(%input : tensor<4xf32>) outs(%init : tensor<3x4x2xf32>) dimensions = [0, 2]
  return %0 : tensor<3x4x2xf32>
}
