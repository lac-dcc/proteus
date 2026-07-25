#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_oracle_soundness as oracle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLIR_OUT_DIR = os.path.join(ROOT, "mlir_out_zerobias")
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

BARE_CALL_TEMPLATE = """  func.func @main() {{
{input_ops}
    %out = call @{func_name}({input_value}) : (tensor<{in_shape}xf32>) -> tensor<{out_shape}xf32>
    return
  }}
}}
"""


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


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(f"error: {name} not found on PATH")
    return path


def hyperfine_time(
    model_path: str,
    mlir_opt: str,
    mlir_runner: str,
    shared_libs: str,
    seed_lattice: str = "",
    rewrite: str | None = None,
    warmup: int = 1,
    runs: int = 5,
) -> float | None:
    hyperfine = require_tool("hyperfine")

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
    wrapped = text[: text.rfind("}")] + BARE_CALL_TEMPLATE.format(
        in_shape=in_shape,
        out_shape=out_shape,
        func_name=func_name,
        input_ops=input_ops,
        input_value=input_value,
    )

    with tempfile.TemporaryDirectory(prefix=f"proteus-hyperfine-{name}-") as tmpdir:
        wrapped_path = os.path.join(tmpdir, "wrapped.mlir")
        lowered_path = os.path.join(tmpdir, "lowered.mlir")
        results_path = os.path.join(tmpdir, "results.json")
        open(wrapped_path, "w").write(wrapped)

        opt = subprocess.run(
            [mlir_opt, wrapped_path, f"--pass-pipeline={LOWERING_PIPELINE}", "-o", lowered_path],
            capture_output=True, text=True,
        )
        if opt.returncode != 0:
            return None

        run_cmd = (
            f"{mlir_runner} {lowered_path} --shared-libs={shared_libs} "
            f"--entry-point-result=void --O3"
        )
        try:
            hf = subprocess.run(
                [
                    hyperfine,
                    "--warmup", str(warmup),
                    "--runs", str(runs),
                    "--export-json", results_path,
                    run_cmd,
                ],
                capture_output=True, text=True, timeout=TIMEOUT * (warmup + runs),
            )
        except subprocess.TimeoutExpired:
            return None
        if hf.returncode != 0:
            return None

        with open(results_path) as f:
            data = json.load(f)
        return data["results"][0]["mean"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="model names (without .mlir); default: all")
    parser.add_argument(
        "--seed-lattice",
        default="",
        help="seed the input tensor per this lattice attr instead of all-dense",
    )
    parser.add_argument(
        "--rewrite",
        choices=["linalg", "scf"],
        default=None,
        help=(
            "apply --spa-rewrite with this target before timing; prints a "
            "single seconds value to stdout instead of updating the cache "
            "(requires exactly one model name)"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="with --rewrite/--seed-lattice, hyperfine warmup runs (default: 1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="with --rewrite/--seed-lattice, exact hyperfine run count (default: 5)",
    )
    args = parser.parse_args()

    mlir_opt = llvm_tool("mlir-opt")
    mlir_runner = llvm_tool("mlir-runner")
    shared_libs = ",".join([llvm_tool("libmlir_runner_utils"), llvm_tool("libmlir_c_runner_utils")])

    if args.rewrite or args.seed_lattice:
        if len(args.names) != 1:
            sys.exit("error: --rewrite/--seed-lattice requires exactly one model name")
        model_path = os.path.join(MLIR_OUT_DIR, f"{args.names[0]}.mlir")
        if not os.path.isfile(model_path):
            sys.exit(f"error: model file not found: {model_path}")
        seconds = hyperfine_time(
            model_path, mlir_opt, mlir_runner, shared_libs,
            seed_lattice=args.seed_lattice, rewrite=args.rewrite,
            warmup=args.warmup, runs=args.runs,
        )
        if seconds is None:
            sys.exit("error: benchmarking failed")
        print(f"{seconds:.6f}")
        return

    if args.names:
        model_paths = [os.path.join(MLIR_OUT_DIR, f"{n}.mlir") for n in args.names]
        missing = [p for p in model_paths if not os.path.isfile(p)]
        if missing:
            sys.exit(f"error: model file(s) not found: {', '.join(missing)}")
    else:
        model_paths = sorted(
            os.path.join(MLIR_OUT_DIR, f) for f in os.listdir(MLIR_OUT_DIR) if f.endswith(".mlir")
        )

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
