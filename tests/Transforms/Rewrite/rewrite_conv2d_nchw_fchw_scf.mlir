// RUN: proteus-opt --spa-analysis="lattice-dump=true" --spa-rewrite="target=scf" %s | FileCheck %s

// CHECK-LABEL: func.func @single_dense_block
func.func @single_dense_block(%arg0: tensor<1x1x5x5xf32>, %arg1: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<1x1x4x4xf32>
  %0 = linalg.conv_2d_nchw_fchw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%arg0, %arg1 : tensor<1x1x5x5xf32>, tensor<1x1x2x2xf32>) outs(%init : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// CHECK: %[[ZEROED:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<1x1x4x4xf32>
// CHECK: scf.for %[[N:.*]] = %c0 to %c1 step %c1 iter_args(%{{.*}} = %[[ZEROED]])
// CHECK:   arith.cmpi sge, %[[N]], %c0
// CHECK:   arith.cmpi slt, %[[N]], %c1
// CHECK:   scf.for %[[F:.*]] = %c0 to %c1 step %c1
// CHECK:     scf.for %[[OH:.*]] = %c0 to %c4 step %c1
// CHECK:       arith.cmpi sge, %[[OH]], %c0
// CHECK:       arith.cmpi slt, %[[OH]], %c2
// CHECK:       scf.for %[[OW:.*]] = %c0 to %c4 step %c1
// CHECK:         scf.if
// CHECK:           scf.for %[[C:.*]] = %c0 to %c1
// CHECK:             scf.for %[[KH:.*]] = %c0 to %c2
// CHECK:               scf.for %[[KW:.*]] = %c0 to %c2
// CHECK:                 %[[IH:.*]] = arith.addi %[[OH]], %[[KH]]
// CHECK:                 %[[IW:.*]] = arith.addi %[[OW]], %[[KW]]
// CHECK:                 tensor.extract %arg0[%[[N]], %[[C]], %[[IH]], %[[IW]]]
// CHECK:                 tensor.extract %arg1[%[[F]], %[[C]], %[[KH]], %[[KW]]]
// CHECK:                 arith.mulf
// CHECK:                 arith.addf
// CHECK:           tensor.insert %{{.*}} into %{{.*}}[%[[N]], %[[F]], %[[OH]], %[[OW]]]
// CHECK:         } else {
// CHECK:           scf.yield
// CHECK:         }

// CHECK-LABEL: func.func @fully_sparse_batch
func.func @fully_sparse_batch(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<1x1x4x4xf32>
  %0 = linalg.conv_2d_nchw_fchw {proteus.lattice = [{size = 1 : i64, words = array<i64: 0>}, {size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 15>}]}
       ins(%arg0, %arg1 : tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32>) outs(%init : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}
// CHECK: %[[ZEROED:.*]] = arith.constant {{.*}}dense<0.000000e+00> : tensor<1x1x4x4xf32>
// CHECK-NOT: scf.for
// CHECK: return %[[ZEROED]]
