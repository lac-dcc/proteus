// RUN: proteus-opt --spa-analysis="lattice-dump=true" %s | FileCheck %s

// CHECK-LABEL: func.func @constants
func.func @constants(%a1 : tensor<32x48x24x13xf32>, %a2 : tensor<32x48x24x13xf32>) -> tensor<3x4x5xf32> {

  // CHECK: words = array<i64: 0>
  // CHECK: words = array<i64: 0>
  // CHECK: words = array<i64: 0>
  // CHECK: words = array<i64: 0>
  %b = arith.constant dense<0.0> : tensor<32x48x24x13xf32>

  // CHECK: words = array<i64: 0>
  // CHECK: words = array<i64: 0>
  // CHECK: words = array<i64: 0>
  %c = arith.constant dense<0.0> : tensor<3x4x5xf32>

  return %c : tensor<3x4x5xf32>
}
