// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=lateral" %s | FileCheck %s

// CHECK-LABEL: func.func @vecmat_test
func.func @vecmat_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}]}
    %lhs : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]},
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 15>}]}
    %rhs : tensor<4x4xf32> {proteus.lattice = [ {size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  // CHECK: linalg.vecmat {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
  %0 = linalg.vecmat ins(%lhs, %rhs : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @vecmat_multiple_A_usage_no_agreement
func.func @vecmat_multiple_A_usage_no_agreement(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]}
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},
    %B : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}]},
    %C : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}]},
    %D : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  %0 = linalg.vecmat ins(%B, %A : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %1 = linalg.vecmat ins(%C, %A : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %2 = linalg.vecmat ins(%D, %A : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: func.func @vecmat_multiple_A_usage_yes_agreement
func.func @vecmat_multiple_A_usage_yes_agreement(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 15>}]}
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},
    %B : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}]},
    %C : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 2>}]},
    %D : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  %0 = linalg.vecmat ins(%B, %A : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %1 = linalg.vecmat ins(%C, %A : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %2 = linalg.vecmat ins(%D, %A : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: func.func @vecmat_multiple_D_usage_no_agreement
func.func @vecmat_multiple_D_usage_no_agreement(
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},
    %B : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 15>}]},
    %C : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 4 : i64, words = array<i64: 15>}]},
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
    %D : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  %0 = linalg.vecmat ins(%D, %A : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %1 = linalg.vecmat ins(%D, %A : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %2 = linalg.vecmat ins(%D, %A : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: func.func @vecmat_multiple_D_usage_yes_agreement
func.func @vecmat_multiple_D_usage_yes_agreement(
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 2>}, {size = 4 : i64, words = array<i64: 15>}]},
    %B : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %C : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 15>}]},
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]}
    %D : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  %0 = linalg.vecmat ins(%D, %A : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %1 = linalg.vecmat ins(%D, %B : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %2 = linalg.vecmat ins(%D, %C : tensor<4xf32>, tensor<4x4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}
