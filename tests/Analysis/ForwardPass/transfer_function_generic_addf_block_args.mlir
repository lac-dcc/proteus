// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @generic_addf_test
func.func @generic_addf_test(
    %lhs  : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %rhs  : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 9>}, {size = 3 : i64, words = array<i64: 1>}]},
    %init : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}]}
) -> tensor<4x3xf32> {
  // CHECK: linalg.generic {{.*}}attrs = {proteus.lattice = [{size = 4 : i64, words = array<i64: 11>}, {size = 3 : i64, words = array<i64: 5>}]}
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<4x3xf32>, tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %add = arith.addf %in, %in_1 : f32
    linalg.yield %add : f32
  } -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @generic_addf_propagating_sparsity_test
func.func @generic_addf_propagating_sparsity_test(
    %lhs  : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 9>}, {size = 3 : i64, words = array<i64: 5>}]},
    %rhs  : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 9>}, {size = 3 : i64, words = array<i64: 1>}]},
    %init : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 3 : i64, words = array<i64: 0>}]}
) -> tensor<4x3xf32> {
  // CHECK: linalg.generic {{.*}}attrs = {proteus.lattice = [{size = 4 : i64, words = array<i64: 9>}, {size = 3 : i64, words = array<i64: 5>}]}
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<4x3xf32>, tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %add = arith.addf %in, %in_1 : f32
    linalg.yield %add : f32
  } -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}
