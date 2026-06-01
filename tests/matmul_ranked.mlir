// RUN: proteus-opt --spa %s | FileCheck %s

// CHECK-LABEL: func.func @matmul
func.func @matmul() -> tensor<3x3xf32> {
  %a = arith.constant dense<[
    [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]
  ]> : tensor<3x4xf32>
  %b = arith.constant dense<[
    [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]
  ]> : tensor<4x3xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<3x3xf32>
  %acc = linalg.fill ins(%cst : f32) outs(%init : tensor<3x3xf32>) -> tensor<3x3xf32>
  %0 = linalg.matmul ins(%a, %b : tensor<3x4xf32>, tensor<4x3xf32>)
                     outs(%acc : tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
