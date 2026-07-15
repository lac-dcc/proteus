// RUN: proteus-opt --add-probe-calls %s | FileCheck %s

// CHECK-LABEL: func.func @no_lattice_info
func.func @no_lattice_info(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %init = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.add ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: linalg.add
// CHECK-NEXT: return

// CHECK-LABEL: func.func @with_lattice_info
func.func @with_lattice_info(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %init0 = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.add {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
       ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%init0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %init1 = tensor.empty() : tensor<2x2xf32>
  %1 = linalg.matmul {proteus.lattice = [{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
       ins(%0, %arg2 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%init1 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: %[[ADD_RES:.*]] = linalg.add
// CHECK-NEXT: probe.observe(%[[ADD_RES]] : tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: %[[MATMUL_RES:.*]] = linalg.matmul
// CHECK-NEXT: probe.observe(%[[MATMUL_RES]] : tensor<2x2xf32>) {opID = 1 : i32, resultID = 0 : i32}
// CHECK-NEXT: probe.report
// CHECK-NEXT: return
