// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @transpose_backward_2d_test
func.func @transpose_backward_2d_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
    %input : tensor<4x3xf32>,
    // CHECK: {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]}
    %init : tensor<3x4xf32>
) -> tensor<3x4xf32> {
  // CHECK: linalg.transpose{{.*}}permutation = [1, 0] {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}]}
  %0 = linalg.transpose {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%input : tensor<4x3xf32>) outs(%init : tensor<3x4xf32>) permutation = [1, 0]
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @transpose_backward_3d_cyclic_test
func.func @transpose_backward_3d_cyclic_test(
    // CHECK: {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}, {size = 5 : i64, words = array<i64: 15>}]}
    %input : tensor<3x4x5xf32>,
    %init : tensor<5x3x4xf32>
) -> tensor<5x3x4xf32> {
  // CHECK: linalg.transpose{{.*}}permutation = [2, 0, 1] {proteus.lattice = [{size = 5 : i64, words = array<i64: 15>}, {size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}]}
  %0 = linalg.transpose {proteus.lattice = [{size = 5 : i64, words = array<i64: 15>}, {size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%input : tensor<3x4x5xf32>) outs(%init : tensor<5x3x4xf32>) permutation = [2, 0, 1]
  return %0 : tensor<5x3x4xf32>
}

// CHECK-LABEL: func.func @transpose_backward_chain
func.func @transpose_backward_chain(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
    %a : tensor<4x3xf32>,
    %init0 : tensor<3x4xf32>,
    %init1 : tensor<4x3xf32>,
    %init2 : tensor<3x4xf32>
) -> tensor<3x4xf32> {
  // CHECK: linalg.transpose{{.*}}permutation = [1, 0] {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}]}
  %t1 = linalg.transpose ins(%a : tensor<4x3xf32>) outs(%init0 : tensor<3x4xf32>) permutation = [1, 0]
  // CHECK: linalg.transpose{{.*}}permutation = [1, 0] {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 3>}]}
  %t2 = linalg.transpose ins(%t1 : tensor<3x4xf32>) outs(%init1 : tensor<4x3xf32>) permutation = [1, 0]
  // CHECK: linalg.transpose{{.*}}permutation = [1, 0] {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}]}
  %t3 = linalg.transpose {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 7>}]} ins(%t2 : tensor<4x3xf32>) outs(%init2 : tensor<3x4xf32>) permutation = [1, 0]
  return %t3 : tensor<3x4xf32>
}
