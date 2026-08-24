#!/usr/bin/env python3
import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_runtime as bench  # noqa: E402
import check_oracle_soundness as oracle  # noqa: E402


def compile_time_for_model(
    model_path: str,
    mlir_opt: str,
    seed_lattice: str = "",
    rewrite: str | None = None,
    warmup: int = 1,
    runs: int = 5,
) -> tuple[float, float, list[float]] | None:
    name = os.path.splitext(os.path.basename(model_path))[0]
    text = open(model_path).read().rstrip()
    m = bench.ENTRY_FUNC_RE.search(text)
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
    wrapped = text[: text.rfind("}")] + bench.BARE_CALL_TEMPLATE.format(
        in_shape=in_shape,
        out_shape=out_shape,
        func_name=func_name,
        input_ops=input_ops,
        input_value=input_value,
    )

    with tempfile.TemporaryDirectory(prefix=f"proteus-compile-{name}-") as tmpdir:
        wrapped_path = os.path.join(tmpdir, "wrapped.mlir")
        lowered_path = os.path.join(tmpdir, "lowered.mlir")
        open(wrapped_path, "w").write(wrapped)

        samples = []
        for i in range(warmup + runs):
            start = time.perf_counter()
            opt = subprocess.run(
                [mlir_opt, wrapped_path, f"--pass-pipeline={bench.LOWERING_PIPELINE}",
                 "-o", lowered_path],
                capture_output=True, text=True,
            )
            elapsed = time.perf_counter() - start
            if opt.returncode != 0:
                return None
            if i >= warmup:
                samples.append(elapsed)

        mean = statistics.mean(samples)
        stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
        return mean, stddev, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="model name (without .mlir)")
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
        "--warmup", type=int, default=1, help="discarded warmup runs (default: 1)"
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="timed run count (default: 5)"
    )
    args = parser.parse_args()

    mlir_opt = bench.llvm_tool("mlir-opt")

    model_path = os.path.join(bench.MLIR_OUT_DIR, f"{args.name}.mlir")
    if not os.path.isfile(model_path):
        sys.exit(f"error: model file not found: {model_path}")

    result = compile_time_for_model(
        model_path, mlir_opt,
        seed_lattice=args.seed_lattice, rewrite=args.rewrite,
        warmup=args.warmup, runs=args.runs,
    )
    if result is None:
        sys.exit("error: compile-time measurement failed")
    mean, stddev, samples = result
    sample_str = ",".join(f"{s:.6f}" for s in samples)
    print(f"{mean:.6f}\t{stddev:.6f}\t{sample_str}")


if __name__ == "__main__":
    main()
