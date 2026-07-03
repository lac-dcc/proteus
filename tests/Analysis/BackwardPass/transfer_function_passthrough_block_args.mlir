// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=backward" %s | FileCheck %s

// CHECK-LABEL: func.func @abs_backward_test
func.func @abs_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xf32>,
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.abs {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.abs ins(%a : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %c = linalg.matmul ins(%b, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @ceil_backward_test
func.func @ceil_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xf32>,
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.ceil {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.ceil ins(%a : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %c = linalg.matmul ins(%b, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @floor_backward_test
func.func @floor_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xf32>,
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.floor {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.floor ins(%a : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %c = linalg.matmul ins(%b, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @negf_backward_test
func.func @negf_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xf32>,
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.negf {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.negf ins(%a : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %c = linalg.matmul ins(%b, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @div_backward_test
func.func @div_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xf32>,
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32>,
    %d : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.div {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.div ins(%a, %d : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %c = linalg.matmul ins(%b, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @div_unsigned_backward_test
func.func @div_unsigned_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xi32>,
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xi32>,
    %d : tensor<4x4xi32>
) -> tensor<4x4xi32> {
  // CHECK: linalg.div_unsigned {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.div_unsigned ins(%a, %d : tensor<4x4xi32>, tensor<4x4xi32>) outs(%init : tensor<4x4xi32>) -> tensor<4x4xi32>
  %c = linalg.matmul ins(%b, %rhs : tensor<4x4xi32>, tensor<4x4xf32>) outs(%init : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %c : tensor<4x4xi32>
}

// CHECK-LABEL: func.func @copy_backward_test
func.func @copy_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xf32>,
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.copy {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.copy ins(%a : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %c = linalg.matmul ins(%b, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @tanh_backward_test
func.func @tanh_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xf32>,
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.tanh {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.tanh ins(%a : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %c = linalg.matmul ins(%b, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @square_backward_test
func.func @square_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xf32>,
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.square {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.square ins(%a : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %c = linalg.matmul ins(%b, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @sqrt_backward_test
func.func @sqrt_backward_test(
    // CHECK: {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
    %a : tensor<4x4xf32>,
    %rhs : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 15>}]},
    %init : tensor<4x4xf32>
) -> tensor<4x4xf32> {
  // CHECK: linalg.sqrt {proteus.lattice = [{size = 4 : i64, words = array<i64: 15>}, {size = 4 : i64, words = array<i64: 7>}]}
  %b = linalg.sqrt ins(%a : tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %c = linalg.matmul ins(%b, %rhs : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}
