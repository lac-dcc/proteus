// RUN: proteus-opt --spa-analysis="pass-stage=seed"     %s | FileCheck %s
// RUN: proteus-opt --spa-analysis="pass-stage=forward"  %s | FileCheck %s
// RUN: proteus-opt --spa-analysis="pass-stage=lateral"  %s | FileCheck %s
// RUN: proteus-opt --spa-analysis="pass-stage=backward" %s | FileCheck %s
// RUN: not proteus-opt --spa-analysis="pass-stage=ltleral" %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

// CHECK-LABEL: func.func @pass_stage_test
// CHECK-ERR: spa-analysis: invalid last-pass value 'ltleral'

func.func @pass_stage_test(
    %a : tensor<2x2xf32>
) -> tensor<2x2xf32> {
  return %a : tensor<2x2xf32>
}
