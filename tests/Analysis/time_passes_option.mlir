// RUN: proteus-opt --spa-analysis="time-passes=true" %s 2>&1 | FileCheck %s

// CHECK: Execution time report
// CHECK: ----Wall Time----  ----Name----
// CHECK: Seed
// CHECK: Forward
// CHECK: Lateral
// CHECK: Backward
// CHECK: Rest
// CHECK: Total

// CHECK-LABEL: func.func @time_passes_test
func.func @time_passes_test(
    %a : tensor<2x2xf32>
) -> tensor<2x2xf32> {
  return %a : tensor<2x2xf32>
}
