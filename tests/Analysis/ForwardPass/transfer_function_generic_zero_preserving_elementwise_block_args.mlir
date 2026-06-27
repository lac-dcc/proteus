// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @generic_mulf_negf_test
func.func @generic_mulf_negf_test(
    %lhs  : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %rhs  : tensor<4x3xf32>,
    %init : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: linalg.generic {{.*}}attrs = {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]}
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<4x3xf32>, tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %mul = arith.mulf %in, %in_1 : f32
    %neg = arith.negf %mul : f32
    linalg.yield %neg : f32
  } -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @generic_non_zero_preserving_test
func.func @generic_non_zero_preserving_test(
    %lhs   : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %rhs   : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %bias  : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %init  : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: linalg.generic {{.*}}attrs = {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs, %bias : tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
    %mul = arith.mulf %in, %in_1 : f32
    %add = arith.addf %mul, %in_2 : f32
    linalg.yield %add : f32
  } -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @generic_mulf_reduction_test
func.func @generic_mulf_reduction_test(
    %input : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %init  : tensor<4xf32>
) -> tensor<4xf32> {
  // CHECK: linalg.generic {{.*}}attrs = {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<4x3xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mul = arith.mulf %in, %out : f32
    %neg = arith.negf %mul : f32
    linalg.yield %neg : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
