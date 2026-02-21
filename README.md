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

* **LLVM/MLIR:** The project is built against LLVM/MLIR.
* **CMake:** Version 3.15 or higher.
* **C++ Compiler:** Supporting C++17 or higher (e.g., GCC 9+, Clang 10+).
* **Ninja/Make:** Build system generators.

## Building Proteus

Follow these standard steps to build the project from the root directory:

```bash
# Create and enter the build directory
mkdir build
cd build

# Configure the project with CMake
# Point to your LLVM/MLIR installation if it's not in the system path
cmake -G Ninja .. \
  -DMLIR_DIR=$LLVM_INSTALL_DIR/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL_DIR/lib/cmake/llvm

# Compile the tools
ninja

```

## Running the Tool

You can run the sparsity analysis pass using the `proteus-opt` tool on MLIR input files:

```bash
./bin/proteus-opt --spa-analysis ../tests/t0.mlir

```

### Example Output

When running the analysis on an einsum-based computational graph, Proteus generates an abstract state  for each tensor. The output includes sparsity vectors (bitmaps) for each dimension.

**Input snippet:**

```mlir
// Matrix Multiplication: C = A(ik) * B(kj)
%C = "proteus.einsum"(%A, %B) {indexing_maps = [ik, kj -> ij]} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

```

**Expected Analysis Output:**

```text
Sparsity Analysis Results:
- Tensor %A: alpha(A) = ([1, 0], [1, 1])  // Row 1 is zero
- Tensor %B: alpha(B) = ([1, 1], [0, 1])  // Column 0 is zero
- Tensor %C: alpha(C) = ([1, 0], [0, 1])  // Resulting sparsity propagated forward/laterally

```

The output `[1, 0]` indicates that the first slice is potentially non-zero, while the second slice is guaranteed to be zero. These bitmaps allow Proteus to optimize the underlying **fiber tree** representations by eliminating zero-valued leaves.