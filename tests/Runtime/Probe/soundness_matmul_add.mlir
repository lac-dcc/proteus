// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s \
// RUN: | grep "linalg.add" \
// RUN: | tee %t.predicted \
// RUN: | FileCheck %s --check-prefix=PREDICTED

// RUN: proteus-opt %s \
// RUN:   --pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries}, \
// RUN:                    convert-linalg-to-loops, \
// RUN:                    probe-lower-to-func-calls, \
// RUN:                    finalize-memref-to-llvm, \
// RUN:                    convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm, \
// RUN:                    func.func(llvm-request-c-wrappers), \
// RUN:                    convert-func-to-llvm, \
// RUN:                    reconcile-unrealized-casts, \
// RUN:                    func.func(canonicalize,cse) \
// RUN:                    )" \
// RUN: | mlir-runner -e chain -entry-point-result=void --shared-libs=%proteus_runtime_lib \
// RUN: | tee %t.actual \
// RUN: | FileCheck %s --check-prefix=ACTUAL

// RUN: python3 %S/check_superset.py %t.predicted %t.actual

func.func @chain() {
  %a = arith.constant dense<[[1.0, 1.0, -1.0],
                             [2.0, 1.0, 3.0],
                             [0.0, 2.0, 1.0],
                             [1.0, -3.0, 2.0]]> : tensor<4x3xf32>
  %b = arith.constant dense<[[1.0, 0.0],
                             [0.0, 1.0],
                             [1.0, 1.0]]> : tensor<3x2xf32>
  %init0 = arith.constant dense<0.0> : tensor<4x2xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x3xf32>, tensor<3x2xf32>)
                     outs(%init0 : tensor<4x2xf32>) -> tensor<4x2xf32>
  %bias = arith.constant dense<[[0.0, 0.0],
                                [1.0, 0.0],
                                [0.0, 2.0],
                                [-3.0, 1.0]]> : tensor<4x2xf32>
  %init1 = arith.constant dense<0.0> : tensor<4x2xf32>
  %e = linalg.add ins(%c, %bias : tensor<4x2xf32>, tensor<4x2xf32>)
                  outs(%init1 : tensor<4x2xf32>) -> tensor<4x2xf32>
  probe.observe(%e: tensor<4x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
  probe.report()
  return
}

// PREDICTED: linalg.add {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 2 : i64, words = array<i64: 3>}]}
// ACTUAL: opID=0 resultID=0 runtime_lattice=[{size = 4 : i64, words = array<i64: 6>}, {size = 2 : i64, words = array<i64: 3>}]
