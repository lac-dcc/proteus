#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ONNX_DIR="${ONNX_DIR:-$REPO_ROOT/external/bennu/models}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/mlir_out}"
ZEROBIAS_DIR="${ZEROBIAS_DIR:-$REPO_ROOT/mlir_out_zerobias}"

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

    echo "Converting $name ..."

    tmp_torch="$(mktemp --suffix=.mlir)"
    import_temp_dir="$(mktemp -d)"
    trap 'rm -f "$tmp_torch"; rm -rf "$import_temp_dir"' EXIT

    # --temp-dir keeps scratch files out of $ONNX_DIR, which may be a
    # read-only mount of the external/bennu submodule's model data.
    torch-mlir-import-onnx --temp-dir="$import_temp_dir" "$onnx_file" -o "$tmp_torch"

    torch-mlir-opt \
        --torch-onnx-to-torch-backend-pipeline \
        --torch-backend-to-linalg-on-tensors-backend-pipeline \
        "$tmp_torch" -o "$linalg_out"

    rm -f "$tmp_torch"
    trap - EXIT

    echo "  -> $linalg_out"
done

splat_weights_flag=()
if [[ -n "${SPLAT_WEIGHTS:-}" ]]; then
    splat_weights_flag=(--splat-weights)
fi
python3 "$SCRIPT_DIR/zero_conv_bias.py" "$OUT_DIR" "$ZEROBIAS_DIR" "${splat_weights_flag[@]}"

rm -f "$OUT_DIR"/*.mlir

echo "Done."
