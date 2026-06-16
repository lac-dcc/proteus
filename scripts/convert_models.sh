#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ONNX_DIR="${ONNX_DIR:-$REPO_ROOT/models}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/mlir_out}"

# Models known not to work with torch-mlir
SKIP="densenet121 densenet161 densenet169 densenet201 vgg13"

mkdir -p "$OUT_DIR"

for onnx_file in "$ONNX_DIR"/*.onnx; do
    name="$(basename "$onnx_file" .onnx)"

    if echo "$SKIP" | grep -qw "$name"; then
        echo "Skipping $name (unsupported)"
        continue
    fi

    linalg_out="$OUT_DIR/${name}.mlir"

    if [[ -f "$linalg_out" ]]; then
        echo "Already converted: $name — skipping (delete $linalg_out to re-run)"
        continue
    fi

    echo "Converting $name ..."

    tmp_torch="$(mktemp --suffix=.mlir)"
    trap 'rm -f "$tmp_torch"' EXIT

    torch-mlir-import-onnx "$onnx_file" -o "$tmp_torch"

    torch-mlir-opt \
        --torch-onnx-to-torch-backend-pipeline \
        --torch-backend-to-linalg-on-tensors-backend-pipeline \
        "$tmp_torch" -o "$linalg_out"

    rm -f "$tmp_torch"
    trap - EXIT

    echo "  -> $linalg_out"
done

echo "Done."
