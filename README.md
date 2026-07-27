<p align="center">
  <img alt="Project Banner" src="assets/images/banner.png" width="90%" height="auto"/></br>
</p>

Proteus is an MLIR-based implementation of **[Sparsity Propagation Analysis](https://homepages.dcc.ufmg.br/~fernando/publications/papers/CGO26_Kaio.pdf) (SPA)**. It provides a static analysis framework to conservatively infer sparsity in tensor slices by propagating information across computational graphs. Unlike traditional element-wise analyses, Proteus operates on dimension-indexed bitmaps, making it asymptotically faster than graph execution.
## Goal of the Analysis

The primary goal of Proteus is to identify **tensor slices** (subsets of elements sharing a fixed subset of indices) that can be safely treated as zero. By recognizing structural sparsity at the compiler level, Proteus enables optimizations that:

* **Reduce Memory Footprint:** Eliminate the need to store slices proven to be zero.


* **Avoid Redundant Computation:** Skip operations where operands are known to be absorbing elements (e.g., zero in multiplication).


* **Increase Precision:** Use **Forward**, **Backward**, and **Lateral** propagation to find sparsity that single-direction analyses miss.


## Dependencies

To build Proteus, you need the following dependencies:

* **LLVM/MLIR:** The project is built against LLVM/MLIR 22.
* **CMake:** Version 3.20 or higher.
* **C++ Compiler:** Supporting C++17 or higher (e.g., GCC 9+, Clang 10+).
* **Ninja/Make:** Build system generators.
* **git-lfs:** Fetches the ONNX models from the `external/bennu` submodule.
* **Docker:** Runs the model conversion and experiment pipelines in containers.

## Building Proteus

The project is built out-of-tree with CMake, and the root `Makefile` wraps the common workflow:

```bash
# Configure + compile (creates build/)
# Point to your LLVM/MLIR installation if it's not in the system path
export LLVM_INSTALL_DIR=/path/to/llvm-22
make build
```

This produces `build/bin/proteus-opt`.

## Running the Tool

`proteus-opt` is a drop-in for `mlir-opt` with the sparsity analysis pass added. Sparsity is seeded via a `proteus.lattice` attribute on function arguments, arguments without one default to fully dense. Run it with `--spa-analysis`, adding `lattice-dump=true` to annotate every op's results with its inferred lattice:

```bash
./build/bin/proteus-opt --spa-analysis='lattice-dump=true' input.mlir
```

### Example Output

**Input** (`input.mlir`) — a `linalg.matmul` whose operands are seeded with `proteus.lattice`:

```mlir
func.func @matmul_example(
    %lhs : tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>},
                                                {size = 3 : i64, words = array<i64: 7>}]},
    %rhs : tensor<3x4xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>},
                                                {size = 4 : i64, words = array<i64: 5>}]},
    %init : tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>},
                                                 {size = 4 : i64, words = array<i64: 0>}]}
) -> tensor<4x4xf32> {
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<4x3xf32>, tensor<3x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
```

**Actual output** of `proteus-opt --spa-analysis='lattice-dump=true' input.mlir`:

```mlir
module {
  func.func @matmul_example(%arg0: tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 3 : i64, words = array<i64: 7>}]}, %arg1: tensor<3x4xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 5>}]}, %arg2: tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}) -> tensor<4x4xf32> {
    %0 = linalg.matmul {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 4 : i64, words = array<i64: 5>}]} ins(%arg0, %arg1 : tensor<4x3xf32>, tensor<3x4xf32>) outs(%arg2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
```

The matmul's result rows inherit `%lhs`'s row sparsity (`words = 5`) and its result columns inherit `%rhs`'s column sparsity (`words = 5`).

**Actual output** of `proteus-opt --spa-analysis='lattice-dump=true' --spa-rewrite input.mlir`:

```mlir
module {
  func.func @matmul_example(%arg0: tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 3 : i64, words = array<i64: 7>}]}, %arg1: tensor<3x4xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 5>}]}, %arg2: tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}) -> tensor<4x4xf32> {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 3] [1, 1] : tensor<4x3xf32> to tensor<1x3xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, 0] [3, 1] [1, 1] : tensor<3x4xf32> to tensor<3x1xf32>
    %extracted_slice_1 = tensor.extract_slice %arg2[0, 0] [1, 1] [1, 1] : tensor<4x4xf32> to tensor<1x1xf32>
    %0 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%extracted_slice_1 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %inserted_slice = tensor.insert_slice %0 into %arg2[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<4x4xf32>
    %extracted_slice_2 = tensor.extract_slice %arg0[0, 0] [1, 3] [1, 1] : tensor<4x3xf32> to tensor<1x3xf32>
    %extracted_slice_3 = tensor.extract_slice %arg1[0, 2] [3, 1] [1, 1] : tensor<3x4xf32> to tensor<3x1xf32>
    %extracted_slice_4 = tensor.extract_slice %arg2[0, 2] [1, 1] [1, 1] : tensor<4x4xf32> to tensor<1x1xf32>
    %1 = linalg.matmul ins(%extracted_slice_2, %extracted_slice_3 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%extracted_slice_4 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %inserted_slice_5 = tensor.insert_slice %1 into %inserted_slice[0, 2] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<4x4xf32>
    %extracted_slice_6 = tensor.extract_slice %arg0[2, 0] [1, 3] [1, 1] : tensor<4x3xf32> to tensor<1x3xf32>
    %extracted_slice_7 = tensor.extract_slice %arg1[0, 0] [3, 1] [1, 1] : tensor<3x4xf32> to tensor<3x1xf32>
    %extracted_slice_8 = tensor.extract_slice %arg2[2, 0] [1, 1] [1, 1] : tensor<4x4xf32> to tensor<1x1xf32>
    %2 = linalg.matmul ins(%extracted_slice_6, %extracted_slice_7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%extracted_slice_8 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %inserted_slice_9 = tensor.insert_slice %2 into %inserted_slice_5[2, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<4x4xf32>
    %extracted_slice_10 = tensor.extract_slice %arg0[2, 0] [1, 3] [1, 1] : tensor<4x3xf32> to tensor<1x3xf32>
    %extracted_slice_11 = tensor.extract_slice %arg1[0, 2] [3, 1] [1, 1] : tensor<3x4xf32> to tensor<3x1xf32>
    %extracted_slice_12 = tensor.extract_slice %arg2[2, 2] [1, 1] [1, 1] : tensor<4x4xf32> to tensor<1x1xf32>
    %3 = linalg.matmul ins(%extracted_slice_10, %extracted_slice_11 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%extracted_slice_12 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %inserted_slice_13 = tensor.insert_slice %3 into %inserted_slice_9[2, 2] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<4x4xf32>
    return %inserted_slice_13 : tensor<4x4xf32>
  }
}
```

### Running the Experiments

Beyond single-file analysis, the repo has an end-to-end benchmark suite that runs SPA over real ONNX vision models (ResNet, VGG, etc.) under several fixed seed sparsity patterns, and cross-checks each prediction against a runtime oracle. It requires `git-lfs` and `Docker` (see [Dependencies](#dependencies)):

```bash
# 1. Fetch the ONNX models (external/bennu submodule) and convert them to MLIR
#    via a Docker container using torch-mlir. Populates models/, mlir_out/,
#    and mlir_out_zerobias/.
make dataset-convert

# 2. Build Proteus in Release mode inside a Docker container and run the
#    benchmark suite over every model in mlir_out_zerobias/.
make experiments
```

Run `make dataset-clean` to remove the converted MLIR models.
