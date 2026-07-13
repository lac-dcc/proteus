// RUN: proteus-opt %s \
// RUN:   --pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries},probe-lower-to-func-calls,finalize-memref-to-llvm,convert-arith-to-llvm,func.func(llvm-request-c-wrappers),convert-func-to-llvm,reconcile-unrealized-casts,func.func(canonicalize,cse))" \
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
// CHECK-DAG: opID=0 resultID=0 observation=1
// CHECK-DAG: opID=1 resultID=0 observation=1
// CHECK-DAG: opID=2 resultID=0 observation=1
