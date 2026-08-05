#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ZEROBIAS_DIR="${ZEROBIAS_DIR:-$REPO_ROOT/mlir_out_zerobias}"

mkdir -p "$ZEROBIAS_DIR"

cat > "$ZEROBIAS_DIR/mbench_conv2d.mlir" <<'EOF'
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @microbench_conv2d(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x54x54xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<64x3x11x11xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %0 = tensor.empty() : tensor<1x64x54x54xf32>
    %broadcasted = linalg.broadcast ins(%cst_2 : tensor<64xf32>) outs(%0 : tensor<1x64x54x54xf32>) dimensions = [0, 2, 3]
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<4> : vector<2xi64>} ins(%arg0, %cst_1 : tensor<1x3x224x224xf32>, tensor<64x3x11x11xf32>) outs(%broadcasted : tensor<1x64x54x54xf32>) -> tensor<1x64x54x54xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x64x54x54xf32>) outs(%0 : tensor<1x64x54x54xf32>) {
    ^bb0(%in: f32, %out: f32):
      %52 = arith.cmpf ugt, %in, %cst : f32
      %53 = arith.select %52, %in, %cst : f32
      linalg.yield %53 : f32
    } -> tensor<1x64x54x54xf32>
    return %2 : tensor<1x64x54x54xf32>
  }
}
EOF

cat > "$ZEROBIAS_DIR/mbench_depthwise_conv2d.mlir" <<'EOF'
module {
  func.func @microbench_depthwise_conv2d(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<3x3x3xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<3xf32>
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x226x226xf32>
    %0 = tensor.empty() : tensor<1x3x224x224xf32>
    %broadcasted = linalg.broadcast ins(%cst_1 : tensor<3xf32>) outs(%0 : tensor<1x3x224x224xf32>) dimensions = [0, 2, 3]
    %1 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %cst_0 : tensor<1x3x226x226xf32>, tensor<3x3x3xf32>) outs(%broadcasted : tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    return %1 : tensor<1x3x224x224xf32>
  }
}
EOF

cat > "$ZEROBIAS_DIR/mbench_pooling_max.mlir" <<'EOF'
module {
  func.func @microbench_pooling_max(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x112x112xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x3x112x112xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x3x112x112xf32>) -> tensor<1x3x112x112xf32>
    %2 = tensor.empty() : tensor<2x2xf32>
    %3 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %2 : tensor<1x3x224x224xf32>, tensor<2x2xf32>) outs(%1 : tensor<1x3x112x112xf32>) -> tensor<1x3x112x112xf32>
    return %3 : tensor<1x3x112x112xf32>
  }
}
EOF

cat > "$ZEROBIAS_DIR/mbench_pooling_sum.mlir" <<'EOF'
module {
  func.func @microbench_pooling_sum(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x112x112xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x226x226xf32>
    %0 = tensor.empty() : tensor<1x3x112x112xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x3x112x112xf32>) -> tensor<1x3x112x112xf32>
    %2 = tensor.empty() : tensor<3x3xf32>
    %3 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %2 : tensor<1x3x226x226xf32>, tensor<3x3xf32>) outs(%1 : tensor<1x3x112x112xf32>) -> tensor<1x3x112x112xf32>
    return %3 : tensor<1x3x112x112xf32>
  }
}
EOF

echo "Wrote 4 microbenchmarks to $ZEROBIAS_DIR"
