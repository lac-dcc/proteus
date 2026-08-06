// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @pad_backward_test
func.func @pad_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
    %a : tensor<4xf32>,
    %init : tensor<4xf32>
) -> (tensor<4xf32>, tensor<6xf32>) {
  %cst = arith.constant 0.0 : f32
  %b = linalg.abs ins(%a : tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  // CHECK: } {proteus.lattice = [{size = 6 : i64, words = array<i64: 30>}]}
  %c = tensor.pad %a low[1] high[1] {
  ^bb0(%i: index):
    tensor.yield %cst : f32
  } : tensor<4xf32> to tensor<6xf32>
  return %b, %c : tensor<4xf32>, tensor<6xf32>
}

// CHECK-LABEL: func.func @concat_backward_test
func.func @concat_backward_test(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}]}
    %a : tensor<2xf32>,
    %d : tensor<2xf32>,
    %init : tensor<2xf32>
) -> (tensor<2xf32>, tensor<4xf32>) {
  %b = linalg.abs ins(%a : tensor<2xf32>) outs(%init : tensor<2xf32>) -> tensor<2xf32>
  // CHECK: tensor.concat dim(0) %arg0, %arg1 {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
  %c = tensor.concat dim(0) %a, %d : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
  return %b, %c : tensor<2xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @expand_shape_backward_test
func.func @expand_shape_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
    %a : tensor<4xf32>,
    %init : tensor<4xf32>
) -> (tensor<4xf32>, tensor<2x2xf32>) {
  %b = linalg.abs ins(%a : tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]} : tensor<4xf32> into tensor<2x2xf32>
  %c = tensor.expand_shape %a [[0, 1]] output_shape [2, 2] : tensor<4xf32> into tensor<2x2xf32>
  return %b, %c : tensor<4xf32>, tensor<2x2xf32>
}

// CHECK-LABEL: func.func @extract_slice_backward_test
func.func @extract_slice_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
    %a : tensor<4xf32>,
    %init : tensor<4xf32>
) -> (tensor<4xf32>, tensor<2xf32>) {
  %b = linalg.abs ins(%a : tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  // CHECK: tensor.extract_slice %arg0[0] [2] [1] {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}]}
  %c = tensor.extract_slice %a[0] [2] [1] : tensor<4xf32> to tensor<2xf32>
  return %b, %c : tensor<4xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @collapse_shape_backward_test
func.func @collapse_shape_backward_test(
    // CHECK: {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
    %a : tensor<2x2xf32>,
    %init : tensor<2x2xf32>
) -> (tensor<2x2xf32>, tensor<4xf32>) {
  %b = linalg.abs ins(%a : tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: tensor.collapse_shape %arg0 {{\[\[}}0, 1]] {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
  %c = tensor.collapse_shape %a [[0, 1]] : tensor<2x2xf32> into tensor<4xf32>
  return %b, %c : tensor<2x2xf32>, tensor<4xf32>
}
