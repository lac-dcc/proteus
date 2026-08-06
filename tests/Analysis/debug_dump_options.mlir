// RUN: proteus-opt --spa-analysis="pass-stage=forward state-dump=true" %s 2>&1 | FileCheck %s --check-prefix=STATE
// RUN: proteus-opt --spa-analysis="pass-stage=forward print-zeros=true" %s 2>&1 | FileCheck %s --check-prefix=ZEROS

// STATE: === SPA State Table ===
// STATE-DAG: <block argument> of type 'tensor<2x3xf32>' at index: 0 (%arg0)
// STATE-DAG: a[0]: 11  (2 slices)
// STATE-DAG: a[1]: 111  (3 slices)
// STATE-DAG: %cst = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
// STATE-DAG: a[0]: 00  (2 slices)
// STATE-DAG: a[1]: 000  (3 slices)
// STATE: ======================

// ZEROS: === Zero Counts ===
// ZEROS-DAG: [block arg #0]
// ZEROS-DAG: dim[0]: 0/2 zeros  dim[1]: 0/3 zeros  | total: 0
// ZEROS-DAG: [arith.constant]
// ZEROS-DAG: dim[0]: 2/2 zeros  dim[1]: 3/3 zeros  | total: 5
// ZEROS: Grand total: 5 zero(s)
// ZEROS: ===================

func.func @debug_dump_test(
    %a : tensor<2x3xf32>
) -> tensor<2x3xf32> {
  %c = arith.constant dense<0.0> : tensor<2x3xf32>
  return %a : tensor<2x3xf32>
}
