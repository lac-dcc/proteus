// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=lateral" %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_test
func.func @matmul_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 7>}]}
    %lhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},

    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]}
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 7>}]},

    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 15>}]}
    %3   : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},

    %init : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4x4xf32> {
  // CHECK: linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 5>}]}
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = linalg.matmul ins(%0, %3 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @matmul_multiple_lhs_usage_no_agreement
func.func @matmul_multiple_lhs_usage_no_agreement(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]}
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},
    %B : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 15>}]},
    %C : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 4 : i64, words = array<i64: 15>}]},
    %D : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4x4xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = linalg.matmul ins(%A, %C : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = linalg.matmul ins(%A, %D : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @matmul_multiple_lhs_usage_yes_agreement
func.func @matmul_multiple_lhs_usage_yes_agreement(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 7>}]}
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},
    %B : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 15>}]},
    %C : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 6>}, {size = 4 : i64, words = array<i64: 15>}]},
    %D : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4x4xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = linalg.matmul ins(%A, %C : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = linalg.matmul ins(%A, %D : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @matmul_multiple_rhs_usage_no_agreement
func.func @matmul_multiple_rhs_usage_no_agreement(
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 5>}]},
    %B : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 10>}]},
    %C : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]},
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 13>}]}
    %D : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 13>}]},
    %init : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4x4xf32> {
  %0 = linalg.matmul ins(%A, %D : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = linalg.matmul ins(%B, %D : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = linalg.matmul ins(%C, %D : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @matmul_multiple_rhs_usage_yes_agreement
func.func @matmul_multiple_rhs_usage_yes_agreement(
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 5>}]},
    %B : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 6>}]},
    %C : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]},
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 13>}]}
    %D : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 13>}]},
    %init : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4x4xf32> {
  %0 = linalg.matmul ins(%A, %D : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = linalg.matmul ins(%B, %D : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = linalg.matmul ins(%C, %D : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
