// RUN: proteus-opt --spa-analysis="lattice-dump=true" --spa-rewrite="target=linalg" %s | FileCheck %s

// CHECK-LABEL: func.func @single_dense_block
func.func @single_dense_block(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<4x4xf32>
  %0 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// CHECK: %[[ZEROED:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<4x4xf32>
// CHECK: %[[LHS:.*]] = tensor.extract_slice %arg0[0, 0] [2, 4]
// CHECK: %[[SUB:.*]] = linalg.matmul ins(%[[LHS]], %arg1
// CHECK: %[[OUT:.*]] = tensor.insert_slice %[[SUB]] into %[[ZEROED]][0, 0]
// CHECK: return %[[OUT]]

// CHECK-LABEL: func.func @fragmented_rows
func.func @fragmented_rows(%arg0: tensor<6x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<6x4xf32> {
  %init = arith.constant dense<0.0> : tensor<6x4xf32>
  %0 = linalg.matmul {proteus.lattice = [{size = 6 : i64, words = array<i64: 51>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%arg0, %arg1 : tensor<6x4xf32>, tensor<4x4xf32>) outs(%init : tensor<6x4xf32>) -> tensor<6x4xf32>
  return %0 : tensor<6x4xf32>
}
// CHECK: %[[ZEROED:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<6x4xf32>
// CHECK: %[[LHS0:.*]] = tensor.extract_slice %arg0[0, 0] [2, 4]
// CHECK: %[[SUB0:.*]] = linalg.matmul ins(%[[LHS0]], %arg1
// CHECK: %[[OUT0:.*]] = tensor.insert_slice %[[SUB0]] into %[[ZEROED]][0, 0]
// CHECK: %[[LHS1:.*]] = tensor.extract_slice %arg0[4, 0] [2, 4]
// CHECK: %[[SUB1:.*]] = linalg.matmul ins(%[[LHS1]], %arg1
// CHECK: %[[OUT1:.*]] = tensor.insert_slice %[[SUB1]] into %[[OUT0]][4, 0]
// CHECK: return %[[OUT1]]

// CHECK-LABEL: func.func @fully_sparse_rows
func.func @fully_sparse_rows(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<4x4xf32>
  %0 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// CHECK: %[[ZEROED:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<4x4xf32>
// CHECK-NOT: linalg.matmul
// CHECK: return %[[ZEROED]]
