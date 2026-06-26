// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @generic_relu_test
func.func @generic_relu_test(
    %input : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]},
    %init  : tensor<4x3xf32>
) -> tensor<4x3xf32> {
  // CHECK: linalg.generic {{.*}}attrs = {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 3 : i64, words = array<i64: 5>}]}
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %cmp = arith.cmpf ogt, %in, %out : f32
    %sel = arith.select %cmp, %in, %out : f32
    linalg.yield %sel : f32
  } -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}
