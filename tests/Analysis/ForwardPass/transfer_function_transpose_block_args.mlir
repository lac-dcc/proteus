// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @transpose_2d_test
func.func @transpose_2d_test(
    %input : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 10>},
                                                  {size = 3 : i64, words = array<i64: 5>}]},
    %init : tensor<3x4xf32>
) -> tensor<3x4xf32> {
  // CHECK: linalg.transpose{{.*}}permutation = [1, 0] {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 10>}]}
  %0 = linalg.transpose ins(%input : tensor<4x3xf32>) outs(%init : tensor<3x4xf32>) permutation = [1, 0]
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @transpose_3d_cyclic_test
func.func @transpose_3d_cyclic_test(
    %input : tensor<3x4x5xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>},
                                                    {size = 4 : i64, words = array<i64: 10>},
                                                    {size = 5 : i64, words = array<i64: 21>}]},
    %init : tensor<5x3x4xf32>
) -> tensor<5x3x4xf32> {
  // CHECK: linalg.transpose{{.*}}permutation = [2, 0, 1] {proteus.lattice = [{size = 5 : i64, words = array<i64: 21>}, {size = 3 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 10>}]}
  %0 = linalg.transpose ins(%input : tensor<3x4x5xf32>) outs(%init : tensor<5x3x4xf32>) permutation = [2, 0, 1]
  return %0 : tensor<5x3x4xf32>
}

// CHECK-LABEL: func.func @transpose_3d_inner_swap_test
func.func @transpose_3d_inner_swap_test(
    %input : tensor<3x4x5xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>},
                                                    {size = 4 : i64, words = array<i64: 10>},
                                                    {size = 5 : i64, words = array<i64: 21>}]},
    %init : tensor<3x5x4xf32>
) -> tensor<3x5x4xf32> {
  // CHECK: linalg.transpose{{.*}}permutation = [0, 2, 1] {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>}, {size = 5 : i64, words = array<i64: 21>}, {size = 4 : i64, words = array<i64: 10>}]}
  %0 = linalg.transpose ins(%input : tensor<3x4x5xf32>) outs(%init : tensor<3x5x4xf32>) permutation = [0, 2, 1]
  return %0 : tensor<3x5x4xf32>
}
