// RUN: proteus-opt %s \
// RUN:   --pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries}, \
// RUN:                    probe-lower-to-func-calls, \
// RUN:                    finalize-memref-to-llvm,convert-arith-to-llvm, \
// RUN:                    func.func(llvm-request-c-wrappers), \
// RUN:                    convert-func-to-llvm, \
// RUN:                    reconcile-unrealized-casts, \
// RUN:                    func.func(canonicalize,cse) \
// RUN:                    )" \
// RUN: | mlir-runner -e foo -entry-point-result=void --shared-libs=%proteus_runtime_lib \
// RUN: | FileCheck %s

func.func @foo() {
  %c0 = arith.constant dense<[[1., 0.], [0., 1.]]> : tensor<2x2xf32>
  probe.observe(%c0: tensor<2x2xf32>) {opID = 0 : i32, resultID = 0 : i32}
  %c1 = arith.constant dense<[[0., 1.], [1., 0.]]> : tensor<2x2xf32>
  probe.observe(%c1: tensor<2x2xf32>) {opID = 1 : i32, resultID = 0 : i32}
  %c2 = arith.constant dense<[[1., 1.], [1., 1.]]> : tensor<2x2xf32>
  probe.observe(%c2: tensor<2x2xf32>) {opID = 2 : i32, resultID = 0 : i32}
  probe.report()
  return
}
// CHECK-DAG: opID=0 resultID=0 runtime_lattice=[{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]
// CHECK-DAG: opID=1 resultID=0 runtime_lattice=[{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]
// CHECK-DAG: opID=2 resultID=0 runtime_lattice=[{size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]
