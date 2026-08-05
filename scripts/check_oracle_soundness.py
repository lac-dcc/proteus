#!/usr/bin/env python3

import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_runtime as bench  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.environ.get("PROTEUS_BUILD_DIR", "build-release")
PROTEUS_OPT = os.path.join(ROOT, BUILD_DIR, "bin", "proteus-opt")
TIMEOUT = 300

WORDS_RE = re.compile(r"words = array<i64:([^>]*)>")
OPID_LINE_RE = re.compile(r"opID=(\d+)\s+resultID=\d+\s+runtime_lattice=(.*)")
DIM_RE = re.compile(
    r"\{size\s*=\s*(\d+)\s*:\s*i64,\s*words\s*=\s*array<i64:([^>]*)>\}"
)

LOWERING_PIPELINE = (
    "builtin.module("
    "func.func(canonicalize, cse),"
    "one-shot-bufferize{bufferize-function-boundaries},"
    "probe-lower-to-func-calls,"
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
    "func.func(llvm-request-c-wrappers),"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts)"
)

CALL_WRAPPER_TEMPLATE = """  func.func @main() {{
{input_ops}
    call @{func_name}({input_value}) : (tensor<{in_shape}xf32>) -> tensor<{out_shape}xf32>
    return
  }}
}}
"""


def parse_words(line):
    """Extract every packed i64 word from all `words = array<i64: ...>`
    groups on a line, in order (a dimension >64 elements wide packs into
    more than one word per group)."""
    return [
        int(w) for group in WORDS_RE.findall(line) for w in group.split(",") if w.strip()
    ]


def cleared_bits_per_dim(seed_lattice):
    """Parse a `proteus.lattice` ArrayAttr string into, per dimension, the set
    of indices whose bit is cleared (i.e. proven structurally sparse)."""
    dims = []
    for size_str, words_str in DIM_RE.findall(seed_lattice):
        size = int(size_str)
        words = [int(w) for w in words_str.split(",") if w.strip()]
        cleared = set()
        for i in range(size):
            word = words[i // 64] if i // 64 < len(words) else 0
            if not (word >> (i % 64)) & 1:
                cleared.add(i)
        dims.append(cleared)
    return dims


def contiguous_ranges(indices):
    """Collapse a set of indices into (start, end)-inclusive contiguous runs."""
    ranges = []
    for idx in sorted(indices):
        if ranges and ranges[-1][1] == idx - 1:
            ranges[-1] = (ranges[-1][0], idx)
        else:
            ranges.append((idx, idx))
    return ranges


def build_seeded_input_ops(in_shape, seed_lattice):
    tensor_ty = f"tensor<{in_shape}xf32>"
    shape = [int(s) for s in in_shape.split("x")]
    lines = [f"    %input_0 = arith.constant dense<1.000000e+00> : {tensor_ty}"]
    cur = "%input_0"
    counter = 0
    for dim, cleared in enumerate(cleared_bits_per_dim(seed_lattice)):
        if dim >= len(shape):
            break
        for start, end in contiguous_ranges(cleared):
            length = end - start + 1
            slice_shape = list(shape)
            slice_shape[dim] = length
            slice_shape_str = "x".join(str(s) for s in slice_shape)
            offsets = ", ".join(
                str(start) if d == dim else "0" for d in range(len(shape))
            )
            sizes = ", ".join(
                str(length) if d == dim else str(shape[d]) for d in range(len(shape))
            )
            strides = ", ".join("1" for _ in shape)

            counter += 1
            zero_name = f"%zero_{counter}"
            next_name = f"%input_{counter}"
            lines.append(
                f"    {zero_name} = arith.constant dense<0.000000e+00> : "
                f"tensor<{slice_shape_str}xf32>"
            )
            lines.append(
                f"    {next_name} = tensor.insert_slice {zero_name} into {cur}"
                f"[{offsets}] [{sizes}] [{strides}] : "
                f"tensor<{slice_shape_str}xf32> into {tensor_ty}"
            )
            cur = next_name

    return "\n".join(lines), cur


def spa_opts(seed_lattice):
    opts = "lattice-dump=true pass-stage=forward"
    if seed_lattice:
        opts += f" seed-lattice='{seed_lattice}'"
    return opts


def probe_runtime_lib():
    lib_dir = os.path.join(ROOT, BUILD_DIR, "lib")
    for name in os.listdir(lib_dir):
        if name.startswith("libProteusProbeRuntime."):
            return os.path.join(lib_dir, name)
    sys.exit(f"error: libProteusProbeRuntime not found under {lib_dir}")


def predicted_lattices(model, seed_lattice):
    result = subprocess.run(
        [
            PROTEUS_OPT,
            "--mlir-elide-elementsattrs-if-larger=1",
            f"--spa-analysis={spa_opts(seed_lattice)}",
            model,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None, result.stderr

    predicted = {}
    op_id = 0
    for line in result.stdout.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("func.func"):
            continue
        if "proteus.lattice" not in line:
            continue
        predicted[op_id] = parse_words(line)
        op_id += 1
    return predicted, None


def actual_lattices(model, seed_lattice):
    text = open(model).read().rstrip()
    match = bench.ENTRY_FUNC_RE.search(text)
    if not match:
        return None, "could not find entry function signature"
    func_name, in_shape, out_shape = match.groups()

    instrumented = subprocess.run(
        [
            PROTEUS_OPT,
            f"--spa-analysis={spa_opts(seed_lattice)}",
            "--add-probe-calls",
            model,
        ],
        capture_output=True,
        text=True,
    )
    if instrumented.returncode != 0:
        return None, instrumented.stderr

    input_ops, input_value = build_seeded_input_ops(in_shape, seed_lattice)
    wrapped = instrumented.stdout[
        : instrumented.stdout.rfind("}")
    ] + CALL_WRAPPER_TEMPLATE.format(
        func_name=func_name,
        in_shape=in_shape,
        out_shape=out_shape,
        input_ops=input_ops,
        input_value=input_value,
    )

    mlir_runner = bench.llvm_tool("mlir-runner")
    shared_libs = ",".join(
        [
            bench.llvm_tool("libmlir_runner_utils"),
            bench.llvm_tool("libmlir_c_runner_utils"),
            probe_runtime_lib(),
        ]
    )

    with tempfile.TemporaryDirectory(prefix="proteus-oracle-") as tmpdir:
        wrapped_path = os.path.join(tmpdir, "wrapped.mlir")
        lowered_path = os.path.join(tmpdir, "lowered.mlir")
        open(wrapped_path, "w").write(wrapped)

        opt = subprocess.run(
            [
                PROTEUS_OPT,
                wrapped_path,
                f"--pass-pipeline={LOWERING_PIPELINE}",
                "-o",
                lowered_path,
            ],
            capture_output=True,
            text=True,
        )
        if opt.returncode != 0:
            return None, opt.stderr

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
            return None, "mlir-runner timed out"
        if run.returncode != 0:
            return None, run.stderr

    actual = {}
    for line in run.stdout.splitlines():
        match = OPID_LINE_RE.search(line)
        if not match:
            continue
        op_id = int(match.group(1))
        actual[op_id] = parse_words(match.group(2))
    return actual, None


def main():
    if len(sys.argv) not in (2, 3):
        print(
            "usage: check_oracle_soundness.py <model.mlir> [seed-lattice-attr]",
            file=sys.stderr,
        )
        return 1
    model = sys.argv[1]
    seed_lattice = sys.argv[2] if len(sys.argv) == 3 else ""

    predicted, err = predicted_lattices(model, seed_lattice)
    if err:
        print(f"error computing predicted lattices: {err}", file=sys.stderr)
        return 1

    actual, err = actual_lattices(model, seed_lattice)
    if err:
        print(f"error computing actual lattices: {err}", file=sys.stderr)
        return 1

    if not actual:
        print("error: no ops were observed at runtime", file=sys.stderr)
        return 1

    ok = True
    for op_id, actual_words in actual.items():
        predicted_words = predicted.get(op_id)
        if predicted_words is None:
            print(f"opID={op_id}: no predicted lattice found", file=sys.stderr)
            ok = False
            continue
        if len(predicted_words) != len(actual_words):
            print(
                f"opID={op_id}: word count mismatch "
                f"(predicted={len(predicted_words)}, actual={len(actual_words)})",
                file=sys.stderr,
            )
            ok = False
            continue

        # TODO

        print(predicted_words)
        print(actual_words)

        for i, (predicted_word, actual_word) in enumerate(
            zip(predicted_words, actual_words)
        ):
            unproven = actual_word & ~predicted_word
            if unproven != 0:
                print(
                    f"opID={op_id} word {i}: soundness violation "
                    f"predicted={predicted_word:#x} actual={actual_word:#x} "
                    f"dense-in-actual-but-not-predicted={unproven:#x}",
                    file=sys.stderr,
                )
                ok = False

    if not ok:
        return 1

    print(f"OK: {len(actual)} observed op(s) sound against prediction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
