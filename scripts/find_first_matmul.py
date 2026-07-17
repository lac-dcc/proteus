#!/usr/bin/env python3

import os
import re
import subprocess
import sys
from pathlib import Path

BUILD_DIR = os.environ.get("PROTEUS_BUILD_DIR", "build-release")
BINARY = Path(__file__).resolve().parent.parent / BUILD_DIR / "bin" / "proteus-opt"

OP_NAME_PATTERN = re.compile(r"=\s*([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)")

MATFAMILY_OPS = {
    "linalg.matmul",
    "linalg.batch_matmul",
    "linalg.matvec",
    "linalg.vecmat",
}


def labeled_lattice_ops(text: str) -> list[str]:
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
            result.append(label)

    return result


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: find_first_matmul.py <model.mlir>", file=sys.stderr)
        return 1
    model = sys.argv[1]

    # The position of the first matmul-family op among lattice-annotated ops is
    # structural: it doesn't depend on the seed lattice, only on op order. Run
    # with the default (fully-dense) seed once per model instead of once per
    # seed-lattice.
    result = subprocess.run(
        [
            str(BINARY),
            "--mlir-elide-elementsattrs-if-larger=1",
            "--spa-analysis=lattice-dump=true",
            model,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR")
        return 0

    ops = labeled_lattice_ops(result.stdout)
    total = len(ops)
    if total == 0:
        print("ERROR")
        return 0

    first_matmul_idx = next((i for i in range(total) if ops[i] in MATFAMILY_OPS), None)
    matmul_pct = (
        f"{100.0 * (first_matmul_idx + 1) / total:.1f}"
        if first_matmul_idx is not None
        else "None"
    )
    print(matmul_pct)
    return 0


if __name__ == "__main__":
    sys.exit(main())
