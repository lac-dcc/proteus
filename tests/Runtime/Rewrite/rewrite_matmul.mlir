// RUN: proteus-opt %s \
// RUN:   --pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries}, \
// RUN:                    convert-linalg-to-loops, \
// RUN:                    convert-scf-to-cf, \
// RUN:                    convert-cf-to-llvm, \
// RUN:                    expand-strided-metadata, \
// RUN:                    lower-affine, \
// RUN:                    finalize-memref-to-llvm, \
// RUN:                    convert-math-to-llvm, \
// RUN:                    convert-arith-to-llvm, \
// RUN:                    convert-index-to-llvm, \
// RUN:                    func.func(llvm-request-c-wrappers), \
// RUN:                    convert-func-to-llvm, \
// RUN:                    reconcile-unrealized-casts \
// RUN:                    )" \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN: | grep -v "base@" > %t.baseline

// RUN: proteus-opt --spa-rewrite="target=linalg" %s \
// RUN: | proteus-opt \
// RUN:   --pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries}, \
// RUN:                    convert-linalg-to-loops, \
// RUN:                    convert-scf-to-cf, \
// RUN:                    convert-cf-to-llvm, \
// RUN:                    expand-strided-metadata, \
// RUN:                    lower-affine, \
// RUN:                    finalize-memref-to-llvm, \
// RUN:                    convert-math-to-llvm, \
// RUN:                    convert-arith-to-llvm, \
// RUN:                    convert-index-to-llvm, \
// RUN:                    func.func(llvm-request-c-wrappers), \
// RUN:                    convert-func-to-llvm, \
// RUN:                    reconcile-unrealized-casts \
// RUN:                    )" \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN: | grep -v "base@" > %t.rewritten

// RUN: diff %t.baseline %t.rewritten

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %a = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]]> : tensor<4x4xf32>
  %b = arith.constant dense<[[1.0, 0.0, 0.0, 1.0],
                             [0.0, 2.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 2.0]]> : tensor<4x4xf32>
  %init = arith.constant dense<0.0> : tensor<4x4xf32>
  %c = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 3>}, {size = 4 : i64, words = array<i64: 11>}]}
       ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %c_buf = bufferization.to_buffer %c : tensor<4x4xf32> to memref<4x4xf32>
  %c_u = memref.cast %c_buf : memref<4x4xf32> to memref<*xf32>
  call @printMemrefF32(%c_u) : (memref<*xf32>) -> ()
  return
}
