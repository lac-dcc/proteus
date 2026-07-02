// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=lateral" %s | FileCheck %s

// CHECK-LABEL: func.func @matvec_test
func.func @matvec_test(

    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 7>}]}
    %lhs : tensor<4x4xf32> {proteus.lattice = [ {size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},

    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]}
    %rhs : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]},

    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  // CHECK: linalg.matvec {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}]}
  %0 = linalg.matvec ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @matvec_multiple_A_usage_no_agreement
func.func @matvec_multiple_A_usage_no_agreement(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]}
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},
    %B : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}]},
    %C : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}]},
    %D : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  %0 = linalg.matvec ins(%A, %B : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %1 = linalg.matvec ins(%A, %C : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %2 = linalg.matvec ins(%A, %D : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: func.func @matvec_multiple_A_usage_yes_agreement
func.func @matvec_multiple_A_usage_yes_agreement(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 7>}]}
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 15>}]},
    %B : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}]},
    %C : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 6>}]},
    %D : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  %0 = linalg.matvec ins(%A, %B : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %1 = linalg.matvec ins(%A, %C : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %2 = linalg.matvec ins(%A, %D : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: func.func @matvec_multiple_D_usage_no_agreement
func.func @matvec_multiple_D_usage_no_agreement(
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 13>}]},
    %B : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 5>}]},
    %C : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 4 : i64, words = array<i64: 10>}]},
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]}
    %D : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  %0 = linalg.matvec ins(%A, %D : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %1 = linalg.matvec ins(%B, %D : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %2 = linalg.matvec ins(%C, %D : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: func.func @matvec_multiple_D_usage_yes_agreement
func.func @matvec_multiple_D_usage_yes_agreement(
    %A : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 13>}, {size = 4 : i64, words = array<i64: 6>}]},
    %B : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 5>}]},
    %C : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>}, {size = 4 : i64, words = array<i64: 7>}]},
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}]}
    %D : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4xf32> {
  %0 = linalg.matvec ins(%A, %D : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %1 = linalg.matvec ins(%B, %D : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  %2 = linalg.matvec ins(%C, %D : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}
