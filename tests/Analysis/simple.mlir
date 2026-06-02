// RUN: proteus-opt %s | FileCheck %s

// CHECK-LABEL: func.func @add
func.func @add(%a: f32, %b: f32) -> f32 {
    %c = arith.addf %a, %b : f32
    return %c : f32
}
