#!/usr/bin/env python3

import re
import sys
from pathlib import Path

CONST_RE = re.compile(
    r"^(\s*)(%[a-zA-Z0-9_]+)\s*=\s*arith\.constant\s+dense<([^>]*)>\s*:\s*(tensor<[0-9]+xf32>)"
)
BROADCAST_RE = re.compile(
    r"^\s*(%[a-zA-Z0-9_]+)\s*=\s*linalg\.broadcast\s+ins\((%[a-zA-Z0-9_]+)\s*:\s*tensor<[0-9]+xf32>\)"
)
CONV_OUTS_RE = re.compile(
    r"linalg\.(?:conv_2d_nchw_fchw|conv_2d_nhwc_hwcf|depthwise_conv_2d_nchw_chw)\b.*?outs\((%[a-zA-Z0-9_]+)"
)


def zero_conv_biases(text: str) -> tuple[str, int]:
    lines = text.splitlines()

    # Map broadcast result name -> bias source SSA name (the `ins` operand).
    broadcast_bias_source = {}
    for line in lines:
        m = BROADCAST_RE.search(line)
        if m:
            broadcast_bias_source[m.group(1)] = m.group(2)

    # Find every conv op's `outs` operand; if it's a broadcast result we know
    # about, mark that broadcast's bias source for zeroing.
    bias_names_to_zero = set()
    for line in lines:
        m = CONV_OUTS_RE.search(line)
        if not m:
            continue
        outs_name = m.group(1)
        if outs_name in broadcast_bias_source:
            bias_names_to_zero.add(broadcast_bias_source[outs_name])

    # Map bias SSA name -> defining arith.constant line index.
    const_def_line = {}
    for i, line in enumerate(lines):
        m = CONST_RE.match(line)
        if m and m.group(2) in bias_names_to_zero:
            const_def_line[m.group(2)] = i

    changed = 0
    for name, idx in const_def_line.items():
        old_line = lines[idx]
        new_line = CONST_RE.sub(
            lambda m: f"{m.group(1)}{m.group(2)} = arith.constant dense<0.000000e+00> : {m.group(4)}",
            old_line,
        )
        if new_line != old_line:
            lines[idx] = new_line
            changed += 1

    missing = bias_names_to_zero - set(const_def_line)
    if missing:
        print(f"  warning: could not find constant def for {missing}", file=sys.stderr)

    return "\n".join(lines) + "\n", changed


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: zero_conv_bias.py <src_dir> <dst_dir>", file=sys.stderr)
        return 1

    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(src_dir.glob("*.mlir")):
        text = f.read_text()
        new_text, changed = zero_conv_biases(text)
        (dst_dir / f.name).write_text(new_text)
        print(f"{f.name}: zeroed {changed} bias constant(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
