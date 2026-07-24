// RUN: proteus-opt --spa-analysis="lattice-dump=true" --spa-rewrite="target=scf" %s | FileCheck %s

// CHECK-LABEL: func.func @single_dense_block
func.func @single_dense_block(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<4x4xf32>
  %0 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// CHECK: %[[ZEROED:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<4x4xf32>
// CHECK: scf.for %[[I:.*]] = %c0 to %c4 step %c1 iter_args(%{{.*}} = %[[ZEROED]])
// CHECK:   arith.cmpi sge, %[[I]], %c0
// CHECK:   arith.cmpi slt, %[[I]], %c2
// CHECK:   scf.for %[[J:.*]] = %c0 to %c4 step %c1
// CHECK:     arith.cmpi sge, %[[J]], %c0
// CHECK:     arith.cmpi slt, %[[J]], %c4
// CHECK:     scf.if
// CHECK:       scf.for %[[K:.*]] = %c0 to %c4
// CHECK:         tensor.extract %arg0[%[[I]], %[[K]]]
// CHECK:         tensor.extract %arg1[%[[K]], %[[J]]]
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK:       tensor.insert %{{.*}} into %{{.*}}[%[[I]], %[[J]]]
// CHECK:     } else {
// CHECK:       scf.yield
// CHECK:     }

// CHECK-LABEL: func.func @fragmented_rows
func.func @fragmented_rows(%arg0: tensor<6x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<6x4xf32> {
  %init = arith.constant dense<0.0> : tensor<6x4xf32>
  %0 = linalg.matmul {proteus.lattice = [{size = 6 : i64, words = array<i64: 51>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%arg0, %arg1 : tensor<6x4xf32>, tensor<4x4xf32>) outs(%init : tensor<6x4xf32>) -> tensor<6x4xf32>
  return %0 : tensor<6x4xf32>
}
// CHECK: %[[ZEROED:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<6x4xf32>
// CHECK: scf.for %[[I:.*]] = %c0 to %c6 step %c1 iter_args(%{{.*}} = %[[ZEROED]])
// CHECK:   arith.cmpi sge, %[[I]], %c0
// CHECK:   arith.cmpi slt, %[[I]], %c2
// CHECK:   arith.cmpi sge, %[[I]], %c4
// CHECK:   arith.cmpi slt, %[[I]], %c6
// CHECK:   arith.ori
// CHECK:   scf.for %[[J:.*]] = %c0 to %c4 step %c1
// CHECK:     scf.if
// CHECK:       scf.for %[[K:.*]] = %c0 to %c4
// CHECK:         tensor.extract %arg0[%[[I]], %[[K]]]
// CHECK:         tensor.extract %arg1[%[[K]], %[[J]]]
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK:       tensor.insert %{{.*}} into %{{.*}}[%[[I]], %[[J]]]
// CHECK:     } else {
// CHECK:       scf.yield
// CHECK:     }

// CHECK-LABEL: func.func @fully_sparse_rows
func.func @fully_sparse_rows(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<4x4xf32>
  %0 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// CHECK: %[[ZEROED:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<4x4xf32>
// CHECK-NOT: scf.for
// CHECK: return %[[ZEROED]]
