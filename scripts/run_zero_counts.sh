#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$ROOT_DIR/build/bin/proteus-opt"
MLIR_DIR="$ROOT_DIR/mlir_out"

if [[ ! -x "$BINARY" ]]; then
  echo "Error: $BINARY not found or not executable. Run 'make build' first." >&2
  exit 1
fi

GREP="grep"
if [[ "$(uname)" == "Darwin" ]]; then
  if ! command -v ggrep &>/dev/null; then
    echo "Error: GNU grep (ggrep) is required on macOS. Install it with: brew install grep" >&2
    exit 1
  fi
  GREP="ggrep"
fi

zeros_for_stage() {
  "$BINARY" "--spa-analysis=print-zeros=true pass-stage=$2" "$1" 2>&1 >/dev/null \
    | "$GREP" -oP 'Grand total: \K[0-9]+'
}

zeros_for_fill_and_pad() {
  "$BINARY" "--spa-analysis=print-zeros=true pass-stage=forward" "$1" 2>&1 >/dev/null \
    | "$GREP" -A1 -E '\[(linalg\.fill|tensor\.pad)\]' \
    | "$GREP" -oP 'total: \K[0-9]+' \
    | awk '{s+=$1} END{print s+0}'
}

printf "%-25s %10s %12s %10s %10s %10s %10s\n" "Model" "Seed" "Fill+Pad" "Forward" "Lateral" "Backward" "Total"
printf "%-25s %10s %12s %10s %10s %10s %10s\n" "-----" "----" "--------" "-------" "-------" "--------" "-----"

for model in "$MLIR_DIR"/*.mlir; do
  name="$(basename "$model" .mlir)"

  after_seed=$(zeros_for_stage     "$model" seed)
  after_forward=$(zeros_for_stage  "$model" forward)
  after_lateral=$(zeros_for_stage  "$model" lateral)
  after_backward=$(zeros_for_stage "$model" backward)
  fill_pad=$(zeros_for_fill_and_pad "$model")

  delta_seed=$((     after_seed ))
  delta_forward=$((  after_forward  - after_seed ))
  delta_lateral=$((  after_lateral  - after_forward ))
  delta_backward=$(( after_backward - after_lateral ))

  printf "%-25s %10s %12s %10s %10s %10s %10s\n" \
    "$name" "$delta_seed" "$fill_pad" "$delta_forward" "$delta_lateral" "$delta_backward" "$after_backward"
done
