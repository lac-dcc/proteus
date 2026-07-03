// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=lateral" %s | FileCheck %s

// CHECK-LABEL: func.func @default_ops_soundness
func.func @default_ops_soundness(
    // CHECK: tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]}
    %lhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},
    // CHECK: tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]},
    %init : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4x4xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %cmp = arith.cmpf ogt, %in, %out : f32
    %sel = arith.select %cmp, %in, %out : f32
    linalg.yield %sel : f32
  } -> tensor<4x4xf32>
  %1 = linalg.matmul ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
