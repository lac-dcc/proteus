#!/usr/bin/env python3
"""Benchmark a single JIT-executed forward pass for each model in mlir_out/,
so it can be compared against SPA analysis time. Results are cached in
scripts/.model_runtimes.tsv, read by run_zero_counts.sh's "Run 1x(s)" column.

Usage:
    python3 scripts/benchmark_runtime.py                 # all models
    python3 scripts/benchmark_runtime.py resnet18 vgg11   # a subset
"""

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLIR_OUT_DIR = os.path.join(ROOT, "mlir_out")
CACHE_FILE = os.path.join(ROOT, "scripts", ".model_runtimes.tsv")
TIMEOUT = 300

LOWERING_PIPELINE = (
    "builtin.module("
    "func.func(canonicalize, cse),"
    "one-shot-bufferize{bufferize-function-boundaries},"
    "convert-linalg-to-loops,"
    "convert-scf-to-cf,"
    "convert-cf-to-llvm,"
    "expand-strided-metadata,"
    "lower-affine,"
    "finalize-memref-to-llvm,"
    "convert-math-to-llvm,"
    "convert-math-to-libm,"
    "convert-arith-to-llvm,"
    "convert-index-to-llvm,"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts)"
)

WRAPPER_TEMPLATE = """  func.func private @nanoTime() -> i64 attributes {{ llvm.emit_c_interface }}
  func.func private @printI64(i64)

  func.func @main() {{
    %input = arith.constant dense<1.000000e+00> : tensor<{in_shape}xf32>
    %t0 = call @nanoTime() : () -> i64
    %out = call @{func_name}(%input) : (tensor<{in_shape}xf32>) -> tensor<{out_shape}xf32>
    %t1 = call @nanoTime() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    call @printI64(%delta) : (i64) -> ()
    return
  }}
}}
"""

ENTRY_FUNC_RE = re.compile(
    r"func\.func @(\w+)\(%arg0:\s*tensor<([\dx]+)xf32>[^)]*\)\s*->\s*tensor<([\dx]+)xf32>"
)


def llvm_tool(name: str) -> str:
    prefix = os.environ.get("LLVM_PREFIX")
    if not prefix:
        for formula in ("llvm@22", "llvm"):
            try:
                prefix = subprocess.run(
                    ["brew", "--prefix", formula], capture_output=True, text=True, check=True
                ).stdout.strip()
                break
            except Exception:
                continue
    if not prefix:
        sys.exit("Could not locate LLVM/MLIR. Set LLVM_PREFIX to your llvm@22 install.")
    for sub, ext in (("bin", ""), ("lib", ".dylib"), ("lib", ".so")):
        path = os.path.join(prefix, sub, name + ext)
        if os.path.exists(path):
            return path
    sys.exit(f"error: {name} not found under {prefix}")


def benchmark_model(model_path: str, mlir_opt: str, mlir_runner: str, shared_libs: str) -> float | None:
    name = os.path.splitext(os.path.basename(model_path))[0]
    text = open(model_path).read().rstrip()
    m = ENTRY_FUNC_RE.search(text)
    if not m:
        return None
    func_name, in_shape, out_shape = m.groups()
    wrapped = text[: text.rfind("}")] + WRAPPER_TEMPLATE.format(
        in_shape=in_shape, out_shape=out_shape, func_name=func_name
    )

    with tempfile.TemporaryDirectory(prefix=f"proteus-bench-{name}-") as tmpdir:
        wrapped_path = os.path.join(tmpdir, "wrapped.mlir")
        lowered_path = os.path.join(tmpdir, "lowered.mlir")
        open(wrapped_path, "w").write(wrapped)

        opt = subprocess.run(
            [mlir_opt, wrapped_path, f"--pass-pipeline={LOWERING_PIPELINE}", "-o", lowered_path],
            capture_output=True, text=True,
        )
        if opt.returncode != 0:
            return None

        try:
            run = subprocess.run(
                [mlir_runner, lowered_path, f"--shared-libs={shared_libs}",
                 "--entry-point-result=void", "--O3"],
                capture_output=True, text=True, timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return None

        m = re.search(r"-?\d+", run.stdout.strip())
        if run.returncode != 0 or not m:
            return None
        return int(m.group(0)) / 1e9


def main():
    names = sys.argv[1:]
    if names:
        model_paths = [os.path.join(MLIR_OUT_DIR, f"{n}.mlir") for n in names]
        missing = [p for p in model_paths if not os.path.isfile(p)]
        if missing:
            sys.exit(f"error: model file(s) not found: {', '.join(missing)}")
    else:
        model_paths = sorted(
            os.path.join(MLIR_OUT_DIR, f) for f in os.listdir(MLIR_OUT_DIR) if f.endswith(".mlir")
        )

    mlir_opt = llvm_tool("mlir-opt")
    mlir_runner = llvm_tool("mlir-runner")
    shared_libs = ",".join([llvm_tool("libmlir_runner_utils"), llvm_tool("libmlir_c_runner_utils")])

    cache = {}
    if os.path.isfile(CACHE_FILE):
        for line in open(CACHE_FILE):
            n, s, *_ = line.rstrip("\n").split("\t")
            cache[n] = s

    for path in model_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        seconds = benchmark_model(path, mlir_opt, mlir_runner, shared_libs)
        if seconds is not None:
            cache[name] = f"{seconds:.6f}"

    with open(CACHE_FILE, "w") as f:
        for name in sorted(cache):
            f.write(f"{name}\t{cache[name]}\n")


if __name__ == "__main__":
    main()
