// RUN: proteus-opt --spa-rewrite %s | FileCheck %s

// CHECK-LABEL: func.func @with_lattice_info
func.func @with_lattice_info(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %init = tensor.empty() : tensor<2x2xf32>
  // CHECK: linalg.add {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
  %0 = linalg.add {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
       ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
