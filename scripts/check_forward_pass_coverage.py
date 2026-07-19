#!/usr/bin/env python3
"""
Check which MLIR ops used in mlir_out_zerobias/ models are handled by ForwardPass.

Ops are classified by where they appear:
  - depth 2 (inside func body): ForwardPass visits these directly
  - depth 3+ (inside linalg.generic / tensor.pad regions): ForwardPass never
  visits these

Exit code 0 if all top-level ops have transfer functions, 1 otherwise.
"""

import re
import sys
from pathlib import Path

# Ops with dedicated visitOp handlers in ForwardPass
DEDICATED_OPS = {
    "linalg.matmul",
    "linalg.add",
    "linalg.matvec",
    "linalg.vecmat",
    "linalg.transpose",
    "linalg.batch_matmul",
    "linalg.fill",
    "linalg.broadcast",
    "linalg.conv_2d",
    "linalg.conv_2d_nchw_fchw",
    "linalg.conv_2d_nhwc_hwcf",
    "linalg.depthwise_conv_2d_nchw_chw",
    "linalg.pooling_nchw_max",
    "linalg.pooling_nchw_sum",
    "tensor.pad",
    "tensor.expand_shape",
    "tensor.empty",
    "tensor.concat",
    "tensor.extract_slice",
    "tensor.collapse_shape",
}

# Ops handled by visitPassthroughOp
PASSTHROUGH_OPS = {
    "linalg.abs",
    "linalg.ceil",
    "linalg.floor",
    "linalg.negf",
    "linalg.div",
    "linalg.div_unsigned",
    "linalg.copy",
    "linalg.tanh",
    "linalg.square",
    "linalg.sqrt",
}

# Ops handled specially before the TypeSwitch
SPECIAL_OPS = {
    "arith.constant",  # handled by SeedPass::seedConstant
}

HANDLED_OPS = DEDICATED_OPS | PASSTHROUGH_OPS | SPECIAL_OPS

# Match dialect.op_name patterns (e.g. linalg.matmul, tensor.pad)
OP_PATTERN = re.compile(r"\b([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)\b")


def extract_ops_by_nesting(path: Path) -> tuple[set[str], set[str]]:
    """
    Parse an MLIR file and classify ops by nesting depth.

    Depth 0 = module level, 1 = inside module, 2 = func body (top-level),
    3+ = inside a nested region (linalg.generic body, tensor.pad body, etc.).

    Region boundaries are detected by lines ending with '{' (open) and lines
    whose first non-whitespace character is '}' (close).

    Returns:
        top_level: ops seen at depth 2 (directly visited by ForwardPass)
        region_only: ops seen only at depth 3+ (inside nested regions)
    """
    top_level: set[str] = set()
    region_internal: set[str] = set()

    depth = 0

    for line in path.read_text().splitlines():
        rstripped = line.rstrip()
        lstripped = rstripped.lstrip()

        # Close region before classifying ops on this line
        if lstripped.startswith("}"):
            depth = max(0, depth - 1)

        for op in OP_PATTERN.findall(rstripped):
            if depth == 2:
                top_level.add(op)
            elif depth >= 3:
                region_internal.add(op)

        # Open region after classifying ops on this line
        if rstripped.endswith("{"):
            depth += 1

    return top_level, region_internal


def main() -> int:
    repo_root = Path(__file__).parent.parent
    mlir_dir = repo_root / "mlir_out_zerobias"

    model_files = sorted(mlir_dir.glob("*.mlir"))
    if not model_files:
        print(f"No .mlir files found in {mlir_dir}")
        return 1

    top_level_map: dict[str, set[str]] = {}

    for mlir_file in model_files:
        ops, _ = extract_ops_by_nesting(mlir_file)
        for op in ops:
            top_level_map.setdefault(op, set()).add(mlir_file.name)

    all_ops = set(top_level_map)

    handled = {op for op in all_ops if op in HANDLED_OPS}
    unhandled = {op for op in all_ops if op not in HANDLED_OPS}

    print(f"Scanned {len(model_files)} models, {len(all_ops)} unique ops\n")

    print("HANDLED (explicit transfer function in ForwardPass):")
    for op in sorted(handled):
        tag = (
            " (passthrough)"
            if op in PASSTHROUGH_OPS
            else " (special)"
            if op in SPECIAL_OPS
            else ""
        )
        print(f"  [+] {op}{tag}")

    print("\nUNHANDLED (need a transfer function):")
    if unhandled:
        for op in sorted(unhandled):
            models = sorted(top_level_map[op])
            print(f"  [!] {op}  ({len(models)} model(s): {', '.join(models)})")
    else:
        print("  (none)")

    n_covered = len(handled)
    print(
        f"\nTop-level coverage: {n_covered}/{len(all_ops)} ops handled, "
        f"{len(unhandled)} need transfer functions"
    )

    return 1 if unhandled else 0


if __name__ == "__main__":
    sys.exit(main())
