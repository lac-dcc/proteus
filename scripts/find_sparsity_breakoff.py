#!/usr/bin/env python3

import os
import re
import subprocess
import sys
from pathlib import Path

BUILD_DIR = os.environ.get("PROTEUS_BUILD_DIR", "build-release")
BINARY = Path(__file__).resolve().parent.parent / BUILD_DIR / "bin" / "proteus-opt"

DIM_PATTERN = re.compile(
    r"\{size\s*=\s*(\d+)\s*:\s*i64,\s*words\s*=\s*array<i64:\s*([^>]*)>\}"
)
OP_NAME_PATTERN = re.compile(r"=\s*([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)")

TRIVIALLY_SPARSE_OPS = {"linalg.fill", "tensor.pad", "arith.constant"}

MATFAMILY_OPS = {
    "linalg.matmul",
    "linalg.batch_matmul",
    "linalg.matvec",
    "linalg.vecmat",
}


def zero_count(line: str) -> int:
    total = 0
    for size_str, words_str in DIM_PATTERN.findall(line):
        size = int(size_str)
        set_bits = 0
        remaining = size
        for word in (int(w) for w in words_str.split(",")):
            if remaining <= 0:
                break
            bits = min(64, remaining)
            word &= 0xFFFFFFFFFFFFFFFF
            if bits < 64:
                word &= (1 << bits) - 1
            set_bits += bin(word).count("1")
            remaining -= bits
        total += size - set_bits
    return total


def labeled_lattice_lines(text: str) -> list[tuple[str, int]]:
    pending_op = None
    result = []
    for line in text.splitlines():
        if line.lstrip().startswith("func.func"):
            label = "func.func (block args)"
        else:
            match = OP_NAME_PATTERN.search(line)
            if match:
                pending_op = match.group(1)
            label = pending_op if pending_op else "?"

        if "proteus.lattice" in line:
            result.append((label, zero_count(line)))

    return result


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: find_sparsity_breakoff.py <model.mlir> <seed-lattice-attr>",
            file=sys.stderr,
        )
        return 1
    model, seed_lattice = sys.argv[1], sys.argv[2]

    spa_opts = "lattice-dump=true"
    if seed_lattice:
        spa_opts += f" seed-lattice='{seed_lattice}'"

    result = subprocess.run(
        [
            str(BINARY),
            "--mlir-elide-elementsattrs-if-larger=1",
            f"--spa-analysis={spa_opts}",
            model,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR,ERROR")
        return 0

    ops = labeled_lattice_lines(result.stdout)
    total = len(ops)
    if total == 0:
        print("ERROR,ERROR")
        return 0

    last_sparse_idx = next(
        (
            i
            for i in range(total - 1, -1, -1)
            if ops[i][1] > 0 and ops[i][0] not in TRIVIALLY_SPARSE_OPS
        ),
        None,
    )
    breakoff_pct = (
        100.0 * (last_sparse_idx + 1) / total if last_sparse_idx is not None else 0.0
    )

    first_matmul_idx = next(
        (i for i in range(total) if ops[i][0] in MATFAMILY_OPS), None
    )
    matmul_pct = (
        f"{100.0 * (first_matmul_idx + 1) / total:.1f}"
        if first_matmul_idx is not None
        else "None"
    )

    print(f"{breakoff_pct:.1f},{matmul_pct}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
