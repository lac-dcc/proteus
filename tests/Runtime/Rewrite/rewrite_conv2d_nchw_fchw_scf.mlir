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

// RUN: proteus-opt --spa-rewrite="target=scf" %s \
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
  %input = arith.constant dense<[[[[0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [1.0, 2.0, 3.0, 4.0],
                                    [5.0, 6.0, 7.0, 8.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0]]]]> : tensor<1x1x6x4xf32>
  %filter = arith.constant dense<[[[[1.0, 1.0],
                                     [1.0, 1.0]]]]> : tensor<1x1x2x2xf32>
  %init = arith.constant dense<0.0> : tensor<1x1x3x3xf32>
  %0 = linalg.conv_2d_nchw_fchw {strides = dense<[2, 1]> : vector<2xi64>,
                                  proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 2>}, {size = 3 : i64, words = array<i64: 7>}]}
       ins(%input, %filter : tensor<1x1x6x4xf32>, tensor<1x1x2x2xf32>) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  %buf = bufferization.to_buffer %0 : tensor<1x1x3x3xf32> to memref<1x1x3x3xf32>
  %u = memref.cast %buf : memref<1x1x3x3xf32> to memref<*xf32>
  call @printMemrefF32(%u) : (memref<*xf32>) -> ()
  return
}
