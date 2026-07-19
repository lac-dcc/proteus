// RUN: proteus-opt --spa-analysis="lattice-dump=true pass-stage=forward" %s | FileCheck %s

// CHECK-LABEL: func.func @forward
func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<1x3x224x224xf32>
  %init = tensor.empty() : tensor<1x3x224x224xf32>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0, %cst : tensor<1x3x224x224xf32>, tensor<1x3x224x224xf32>) outs(%init : tensor<1x3x224x224xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<1x3x224x224xf32>
  return %0 : tensor<1x3x224x224xf32>
}
