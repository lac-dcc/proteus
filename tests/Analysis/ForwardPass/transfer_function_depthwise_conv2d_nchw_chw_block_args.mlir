// RUN: proteus-opt --spa-analysis="lattice-dump=true" %s | FileCheck %s

// CHECK-LABEL: func.func @depthwise_conv2d_nchw_chw_filter_channel_sparsity
func.func @depthwise_conv2d_nchw_chw_filter_channel_sparsity(
    %input  : tensor<1x3x3x3xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 7>}]},
    %filter : tensor<3x2x2xf32>   {proteus.lattice = [{size = 3 : i64, words = array<i64: 5>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init   : tensor<1x3x2x2xf32>
) -> tensor<1x3x2x2xf32> {
  // CHECK: linalg.depthwise_conv_2d_nchw_chw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 5>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
  %0 = linalg.depthwise_conv_2d_nchw_chw ins(%input, %filter : tensor<1x3x3x3xf32>, tensor<3x2x2xf32>) outs(%init : tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %0 : tensor<1x3x2x2xf32>
}

// CHECK-LABEL: func.func @depthwise_conv2d_nchw_chw_input_channel_sparsity
func.func @depthwise_conv2d_nchw_chw_input_channel_sparsity(
    %input  : tensor<1x3x3x3xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 5>}, {size = 3 : i64, words = array<i64: 7>}, {size = 3 : i64, words = array<i64: 7>}]},
    %filter : tensor<3x2x2xf32>   {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init   : tensor<1x3x2x2xf32>
) -> tensor<1x3x2x2xf32> {
  // CHECK: linalg.depthwise_conv_2d_nchw_chw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 5>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]}
  %0 = linalg.depthwise_conv_2d_nchw_chw ins(%input, %filter : tensor<1x3x3x3xf32>, tensor<3x2x2xf32>) outs(%init : tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %0 : tensor<1x3x2x2xf32>
}

// CHECK-LABEL: func.func @depthwise_conv2d_nchw_chw_h_sparsity
func.func @depthwise_conv2d_nchw_chw_h_sparsity(
    %input  : tensor<1x1x4x4xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 4 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}]},
    %filter : tensor<1x2x2xf32>   {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init   : tensor<1x1x3x3xf32>
) -> tensor<1x1x3x3xf32> {
  // CHECK: linalg.depthwise_conv_2d_nchw_chw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 6>}, {size = 3 : i64, words = array<i64: 7>}]}
  %0 = linalg.depthwise_conv_2d_nchw_chw ins(%input, %filter : tensor<1x1x4x4xf32>, tensor<1x2x2xf32>) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

// CHECK-LABEL: func.func @depthwise_conv2d_nchw_chw_stride2
func.func @depthwise_conv2d_nchw_chw_stride2(
    %input  : tensor<1x1x6x4xf32> {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 6 : i64, words = array<i64: 12>}, {size = 4 : i64, words = array<i64: 15>}]},
    %filter : tensor<1x2x2xf32>   {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 2 : i64, words = array<i64: 3>}, {size = 2 : i64, words = array<i64: 3>}]},
    %init   : tensor<1x1x3x3xf32>
) -> tensor<1x1x3x3xf32> {
  // CHECK: linalg.depthwise_conv_2d_nchw_chw {proteus.lattice = [{size = 1 : i64, words = array<i64: 1>}, {size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 2>}, {size = 3 : i64, words = array<i64: 7>}], strides = dense<[2, 1]> : vector<2xi64>}
  %0 = linalg.depthwise_conv_2d_nchw_chw {strides = dense<[2, 1]> : vector<2xi64>} ins(%input, %filter : tensor<1x1x6x4xf32>, tensor<1x2x2xf32>) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}
