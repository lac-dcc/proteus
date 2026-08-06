// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @fill_unhandled_op_backward_test
func.func @fill_unhandled_op_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
    %a : tensor<4xf32>,
    %init : tensor<4xf32>
) -> (tensor<4xf32>, tensor<4xf32>) {
  %cst = arith.constant 1.0 : f32
  %b = linalg.abs ins(%a : tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  // CHECK: linalg.fill {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
  %d = linalg.fill ins(%cst : f32) outs(%a : tensor<4xf32>) -> tensor<4xf32>
  return %b, %d : tensor<4xf32>, tensor<4xf32>
}
