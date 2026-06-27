// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @generic_clamp_test
func.func @generic_clamp_test(
    %input : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %lower : tensor<f32>,
    %upper : tensor<f32>,
    %init  : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: linalg.generic {{.*}}attrs = {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]}
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %lower, %upper : tensor<4x3xf32>, tensor<f32>, tensor<f32>)
    outs(%init : tensor<4x3xf32>) {
  ^bb0(%in: f32, %lo: f32, %hi: f32, %out: f32):
    %cmp0 = arith.cmpf ult, %in, %lo : f32
    %sel0 = arith.select %cmp0, %lo, %in : f32
    %cmp1 = arith.cmpf ugt, %sel0, %hi : f32
    %sel1 = arith.select %cmp1, %hi, %sel0 : f32
    linalg.yield %sel1 : f32
  } -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}
