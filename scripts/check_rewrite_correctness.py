#!/usr/bin/env python3
"""Checks that --spa-rewrite (both the linalg and scf targets) preserves the
model's output, when compared to the non-transformed IR

Usage:
    check_rewrite_correctness.py <model.mlir> [seed-lattice-attr]
"""

import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_runtime as bench  # noqa: E402
import check_oracle_soundness as oracle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.environ.get("PROTEUS_BUILD_DIR", "build-release")
PROTEUS_OPT = os.path.join(ROOT, BUILD_DIR, "bin", "proteus-opt")
TIMEOUT = 300

REWRITE_TARGETS = ("linalg", "scf")

REWRITE_COUNTS_RE = re.compile(
    r"spa-rewrite counts for @\w+: linalg\.matmul=(\d+), "
    r"linalg\.conv_2d_nchw_fchw=(\d+)"
)

PRINT_WRAPPER_TEMPLATE = """  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {{
{input_ops}
    %out = call @{func_name}({input_value}) : (tensor<{in_shape}xf32>) -> tensor<{out_shape}xf32>
    %out_buf = bufferization.to_buffer %out : tensor<{out_shape}xf32> to memref<{out_shape}xf32>
    %out_u = memref.cast %out_buf : memref<{out_shape}xf32> to memref<*xf32>
    call @printMemrefF32(%out_u) : (memref<*xf32>) -> ()
    return
  }}
}}
"""


def printed_output(model, seed_lattice, rewrite=None):
    text = open(model).read().rstrip()
    match = bench.ENTRY_FUNC_RE.search(text)
    if not match:
        return None, "could not find entry function signature", None
    func_name, in_shape, out_shape = match.groups()

    rewrite_count = None
    if rewrite:
        rewritten = subprocess.run(
            [
                PROTEUS_OPT,
                f"--spa-analysis={oracle.spa_opts(seed_lattice)}",
                f"--spa-rewrite=target={rewrite} count-rewrites=true",
                model,
            ],
            capture_output=True,
            text=True,
        )
        if rewritten.returncode != 0:
            return None, rewritten.stderr, None
        text = rewritten.stdout.rstrip()

        counts_match = REWRITE_COUNTS_RE.search(rewritten.stderr)
        rewrite_count = (
            sum(int(g) for g in counts_match.groups()) if counts_match else 0
        )

    input_ops, input_value = oracle.build_seeded_input_ops(in_shape, seed_lattice)
    wrapped = text[: text.rfind("}")] + PRINT_WRAPPER_TEMPLATE.format(
        func_name=func_name,
        in_shape=in_shape,
        out_shape=out_shape,
        input_ops=input_ops,
        input_value=input_value,
    )

    mlir_runner = bench.llvm_tool("mlir-runner")
    shared_libs = ",".join(
        [bench.llvm_tool("libmlir_runner_utils"), bench.llvm_tool("libmlir_c_runner_utils")]
    )

    with tempfile.TemporaryDirectory(prefix="proteus-correctness-") as tmpdir:
        wrapped_path = os.path.join(tmpdir, "wrapped.mlir")
        lowered_path = os.path.join(tmpdir, "lowered.mlir")
        open(wrapped_path, "w").write(wrapped)

        opt = subprocess.run(
            [
                PROTEUS_OPT,
                wrapped_path,
                f"--pass-pipeline={bench.LOWERING_PIPELINE}",
                "-o",
                lowered_path,
            ],
            capture_output=True,
            text=True,
        )
        if opt.returncode != 0:
            return None, opt.stderr, None

        try:
            run = subprocess.run(
                [
                    mlir_runner,
                    lowered_path,
                    f"--shared-libs={shared_libs}",
                    "-e",
                    "main",
                    "--entry-point-result=void",
                    bench.mlir_runner_opt_flag(),
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return None, "mlir-runner timed out", None
        if run.returncode != 0:
            return None, run.stderr, None

    filtered = "\n".join(
        line for line in run.stdout.splitlines() if "base@" not in line
    )
    return filtered, None, rewrite_count


def main():
    if len(sys.argv) not in (2, 3):
        print(
            "usage: check_rewrite_correctness.py <model.mlir> [seed-lattice-attr]",
            file=sys.stderr,
        )
        return 1
    model = sys.argv[1]
    seed_lattice = sys.argv[2] if len(sys.argv) == 3 else ""

    baseline, err, _ = printed_output(model, seed_lattice)
    if err:
        print(f"error computing baseline output: {err}", file=sys.stderr)
        return 1

    ok = True
    for target in REWRITE_TARGETS:
        rewritten, err, rewrite_count = printed_output(model, seed_lattice, rewrite=target)
        if err:
            print(f"{target}: FAIL")
            print(f"{target}: error computing rewritten output: {err}", file=sys.stderr)
            ok = False
            continue
        if rewrite_count == 0:
            print(
                f"{target}: warning: rewrite pattern rewrote 0 ops "
                "(comparison did not exercise sparsity-skipping logic)",
                file=sys.stderr,
            )
        if rewritten == baseline:
            print(f"{target}: PASS")
        else:
            print(f"{target}: FAIL")
            print(f"{target}: output mismatch", file=sys.stderr)
            print(f"--- baseline\n{baseline}\n--- {target}\n{rewritten}", file=sys.stderr)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
