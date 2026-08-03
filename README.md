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

**Actual output** of `proteus-opt --spa-analysis='lattice-dump=true' --spa-rewrite="target=scf" input.mlir`:

```mlir
module {
  func.func @matmul_example(%arg0: tensor<4x3xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 5>}, {size = 3 : i64, words = array<i64: 7>}]}, %arg1: tensor<3x4xf32> {proteus.lattice = [{size = 3 : i64, words = array<i64: 7>}, {size = 4 : i64, words = array<i64: 5>}]}, %arg2: tensor<4x4xf32> {proteus.lattice = [{size = 4 : i64, words = array<i64: 0>}, {size = 4 : i64, words = array<i64: 0>}]}) -> tensor<4x4xf32> {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (tensor<4x4xf32>) {
      %1 = arith.cmpi sge, %arg3, %c0 : index
      %2 = arith.cmpi slt, %arg3, %c1 : index
      %3 = arith.andi %1, %2 : i1
      %4 = arith.cmpi sge, %arg3, %c2 : index
      %5 = arith.cmpi slt, %arg3, %c3 : index
      %6 = arith.andi %4, %5 : i1
      %7 = arith.ori %3, %6 : i1
      %8 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4x4xf32>) {
        %9 = arith.cmpi sge, %arg5, %c0 : index
        %10 = arith.cmpi slt, %arg5, %c1 : index
        %11 = arith.andi %9, %10 : i1
        %12 = arith.cmpi sge, %arg5, %c2 : index
        %13 = arith.cmpi slt, %arg5, %c3 : index
        %14 = arith.andi %12, %13 : i1
        %15 = arith.ori %11, %14 : i1
        %16 = arith.andi %7, %15 : i1
        %17 = scf.if %16 -> (tensor<4x4xf32>) {
          %extracted = tensor.extract %arg6[%arg3, %arg5] : tensor<4x4xf32>
          %18 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %extracted) -> (f32) {
            %extracted_0 = tensor.extract %arg0[%arg3, %arg7] : tensor<4x3xf32>
            %extracted_1 = tensor.extract %arg1[%arg7, %arg5] : tensor<3x4xf32>
            %19 = arith.mulf %extracted_0, %extracted_1 : f32
            %20 = arith.addf %arg8, %19 : f32
            scf.yield %20 : f32
          }
          %inserted = tensor.insert %18 into %arg6[%arg3, %arg5] : tensor<4x4xf32>
          scf.yield %inserted : tensor<4x4xf32>
        } else {
          scf.yield %arg6 : tensor<4x4xf32>
        }
        scf.yield %17 : tensor<4x4xf32>
      }
      scf.yield %8 : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }
}
```

### Running the Experiments

Beyond single-file analysis, the repo has an end-to-end benchmark suite that runs SPA over real ONNX vision models (ResNet, VGG, etc.) under several fixed seed sparsity patterns. It requires `git-lfs` and `Docker` (see [Dependencies](#dependencies)):

```bash
# 1. Fetch the ONNX models (external/bennu submodule) and convert them to MLIR
#    via a Docker container using torch-mlir. Populates models/, mlir_out/,
#    and mlir_out_zerobias/.
make dataset-convert

# 2. Build Proteus in Release mode inside a Docker container and run the
#    zero count script over every model in mlir_out_zerobias/ across different lattices.
make zero-counts

# 3. Build Proteus in Release mode inside a Docker container and run the
#    timings benchmark over every model in mlir_out_zerobias/ across different lattices.
make timings

# 4. Build Proteus in Release mode inside a Docker container and run the
#    oracles checker script over every model in mlir_out_zerobias/ across different lattices.
make oracles
```

To splat the constants within the dataset for faster conversions as well as lexing and parsing, you can use the `SPLAT_WEIGHTS` variable like so:

```bash
make dataset-convert SPLAT_WEIGHTS=1
```

To filter a benchmark on specific lattices you can use the `LATTICE_FILTER` variable like so:

```bash
# make <benchmark-option> LATTICE_FILTER=banded-64,all-sparse
make oracles LATTICE_FILTER=banded-64,all-sparse
```

Run `make dataset-clean` to remove the converted MLIR models.
