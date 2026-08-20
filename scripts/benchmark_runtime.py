#!/usr/bin/env python3

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_oracle_soundness as oracle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLIR_OUT_DIR = os.path.join(ROOT, "mlir_out_zerobias")

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

ENTRY_FUNC_RE = re.compile(
    r"func\.func @(\w+)\(%arg0:\s*tensor<([\dx]+)xf32>[^)]*\)\s*->\s*tensor<([\dx]+)xf32>"
)

BARE_CALL_TEMPLATE = """  func.func @main() {{
{input_ops}
    %out = call @{func_name}({input_value}) : (tensor<{in_shape}xf32>) -> tensor<{out_shape}xf32>
    return
  }}
}}
"""

NANOTIME_LOOP_TEMPLATE = """  func.func private @nanoTime() -> i64 attributes {{ llvm.emit_c_interface }}
  func.func private @printI64(i64)
  func.func private @printNewline()

  func.func @main() {{
{input_ops}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {iterations} : index
    scf.for %i = %c0 to %cN step %c1 {{
      %t0 = func.call @nanoTime() : () -> i64
      %out = func.call @{func_name}({input_value}) : (tensor<{in_shape}xf32>) -> tensor<{out_shape}xf32>
      %t1 = func.call @nanoTime() : () -> i64
      %delta = arith.subi %t1, %t0 : i64
      func.call @printI64(%delta) : (i64) -> ()
      func.call @printNewline() : () -> ()
    }}
    return
  }}
}}
"""


def mlir_runner_opt_flag() -> str:
    return "--O0" if os.environ.get("NO_O3") else "--O3"


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


def timed_runs(
    model_path: str,
    mlir_opt: str,
    mlir_runner: str,
    shared_libs: str,
    seed_lattice: str = "",
    rewrite: str | None = None,
    warmup: int = 1,
    runs: int = 5,
) -> tuple[float, float, list[float]] | None:
    name = os.path.splitext(os.path.basename(model_path))[0]
    text = open(model_path).read().rstrip()
    m = ENTRY_FUNC_RE.search(text)
    if not m:
        return None
    func_name, in_shape, out_shape = m.groups()

    if rewrite:
        rewritten = subprocess.run(
            [
                oracle.PROTEUS_OPT,
                f"--spa-analysis={oracle.spa_opts(seed_lattice)}",
                f"--spa-rewrite=target={rewrite}",
                model_path,
            ],
            capture_output=True,
            text=True,
        )
        if rewritten.returncode != 0:
            return None
        text = rewritten.stdout.rstrip()

    input_ops, input_value = oracle.build_seeded_input_ops(in_shape, seed_lattice)
    wrapped = text[: text.rfind("}")] + NANOTIME_LOOP_TEMPLATE.format(
        in_shape=in_shape,
        out_shape=out_shape,
        func_name=func_name,
        input_ops=input_ops,
        input_value=input_value,
        iterations=warmup + runs,
    )

    with tempfile.TemporaryDirectory(prefix=f"proteus-timed-{name}-") as tmpdir:
        wrapped_path = os.path.join(tmpdir, "wrapped.mlir")
        lowered_path = os.path.join(tmpdir, "lowered.mlir")
        open(wrapped_path, "w").write(wrapped)

        opt = subprocess.run(
            [mlir_opt, wrapped_path, f"--pass-pipeline={LOWERING_PIPELINE}", "-o", lowered_path],
            capture_output=True, text=True,
        )
        if opt.returncode != 0:
            return None

        run = subprocess.run(
            [mlir_runner, lowered_path, f"--shared-libs={shared_libs}",
             "--entry-point-result=void", mlir_runner_opt_flag()],
            capture_output=True, text=True,
        )
        deltas = [int(v) for v in re.findall(r"-?\d+", run.stdout)]
        if run.returncode != 0 or len(deltas) != warmup + runs:
            return None
        samples = [d / 1e9 for d in deltas[warmup:]]

        mean = statistics.mean(samples)
        stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
        return mean, stddev, samples


def ttest(samples_a: list[float], samples_b: list[float]) -> float | None:
    if len(samples_a) < 2 or len(samples_b) < 2:
        return None
    return stats.ttest_ind(samples_a, samples_b, equal_var=False).pvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", help="model name (without .mlir)")
    parser.add_argument(
        "--seed-lattice",
        default="",
        help="seed the input tensor per this lattice attr instead of all-dense",
    )
    parser.add_argument(
        "--rewrite",
        choices=["linalg", "scf"],
        default=None,
        help="apply --spa-rewrite with this target before timing",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="discarded warmup runs (default: 1)"
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="timed run count (default: 5)"
    )
    parser.add_argument(
        "--pvalue",
        nargs=2,
        metavar=("SAMPLES_A", "SAMPLES_B"),
        help="print a Welch's t-test p-value for two comma-separated sample lists and exit",
    )
    args = parser.parse_args()

    if args.pvalue:
        samples_a = [float(v) for v in args.pvalue[0].split(",") if v]
        samples_b = [float(v) for v in args.pvalue[1].split(",") if v]
        p = ttest(samples_a, samples_b)
        print("n/a" if p is None else f"{p:.4g}")
        return

    if not args.name:
        parser.error("the following arguments are required: name")

    mlir_opt = llvm_tool("mlir-opt")
    mlir_runner = llvm_tool("mlir-runner")
    shared_libs = ",".join([llvm_tool("libmlir_runner_utils"), llvm_tool("libmlir_c_runner_utils")])

    model_path = os.path.join(MLIR_OUT_DIR, f"{args.name}.mlir")
    if not os.path.isfile(model_path):
        sys.exit(f"error: model file not found: {model_path}")

    result = timed_runs(
        model_path, mlir_opt, mlir_runner, shared_libs,
        seed_lattice=args.seed_lattice, rewrite=args.rewrite,
        warmup=args.warmup, runs=args.runs,
    )
    if result is None:
        sys.exit("error: benchmarking failed")
    mean, stddev, samples = result
    sample_str = ",".join(f"{s:.6f}" for s in samples)
    print(f"{mean:.6f}\t{stddev:.6f}\t{sample_str}")


if __name__ == "__main__":
    main()
