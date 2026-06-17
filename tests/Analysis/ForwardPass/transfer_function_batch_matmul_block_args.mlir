// RUN: proteus-opt --spa-analysis="lattice-dump=true" %s | FileCheck %s

// CHECK-LABEL: func.func @batch_matmul_test
func.func @batch_matmul_test(

    %lhs : tensor<3x2x2xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 3>},
                                                 {size = 2 : i64, words = array<i64: 2>},
                                                 {size = 2 : i64, words = array<i64: 3>}]},

    %rhs : tensor<3x2x2xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>},
                                                 {size = 2 : i64, words = array<i64: 3>},
                                                 {size = 2 : i64, words = array<i64: 2>}]},

    %init : tensor<3x2x2xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 0>},
                                                  {size = 2 : i64, words = array<i64: 0>},
                                                  {size = 2 : i64, words = array<i64: 0>}]}

) -> tensor<3x2x2xf32> {
  // CHECK: linalg.batch_matmul {proteus.lattice = [{size = 3 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 2>}, {size = 2 : i64, words = array<i64: 2>}]}
  %0 = linalg.batch_matmul ins(%lhs, %rhs : tensor<3x2x2xf32>, tensor<3x2x2xf32>) outs(%init : tensor<3x2x2xf32>) -> tensor<3x2x2xf32>
  return %0 : tensor<3x2x2xf32>
}
