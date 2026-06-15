// RUN: proteus-opt --spa-analysis="lattice-dump=true" %s | FileCheck %s

// CHECK-LABEL: func.func @constants
func.func @constants(%a1 : tensor<32x48x24x13xf32>, %a2 : tensor<32x48x24x13xf32>) -> tensor<3x4x5xf32> {

  // CHECK: words = array<i64: 4294967295>
  // CHECK: words = array<i64: 281474976710655>
  // CHECK: words = array<i64: 16777215>
  // CHECK: words = array<i64: 8191>
  %b = arith.constant dense<0.0> : tensor<32x48x24x13xf32>

  // CHECK: words = array<i64: 7>
  // CHECK: words = array<i64: 15>
  // CHECK: words = array<i64: 31>
  %c = arith.constant dense<0.0> : tensor<3x4x5xf32>

  return %c : tensor<3x4x5xf32>
}
