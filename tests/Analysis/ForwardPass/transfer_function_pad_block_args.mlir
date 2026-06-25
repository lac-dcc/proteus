// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @pad_1d_symmetric
func.func @pad_1d_symmetric(
    %src  : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}]},
    %init : tensor<6xf32>
) -> tensor<6xf32> {
  %cst = arith.constant 0.0 : f32
  // CHECK: } {proteus.lattice = [{size = 6 : i64, words = array<i64: 20>}]}
  %0 = tensor.pad %src low[1] high[1] {
  ^bb0(%i: index):
    tensor.yield %cst : f32
  } : tensor<4xf32> to tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func.func @pad_1d_low_only
func.func @pad_1d_low_only(
    %src  : tensor<3xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}]},
    %init : tensor<5xf32>
) -> tensor<5xf32> {
  %cst = arith.constant 0.0 : f32
  // CHECK: } {proteus.lattice = [{size = 5 : i64, words = array<i64: 28>}]}
  %0 = tensor.pad %src low[2] high[0] {
  ^bb0(%i: index):
    tensor.yield %cst : f32
  } : tensor<3xf32> to tensor<5xf32>
  return %0 : tensor<5xf32>
}

// CHECK-LABEL: func.func @pad_2d_last_dim_only
func.func @pad_2d_last_dim_only(
    %src  : tensor<3x3xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>}, {size = 3 : i64, words = array<i64: 6>}]},
    %init : tensor<3x5xf32>
) -> tensor<3x5xf32> {
  %cst = arith.constant 0.0 : f32
  // CHECK: } {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>}, {size = 5 : i64, words = array<i64: 12>}]}
  %0 = tensor.pad %src low[0, 1] high[0, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<3x3xf32> to tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}
