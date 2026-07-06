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

# A single full (pass-stage=backward) run with time-passes=true nests a
# TimingScope per stage, so one invocation yields wall-clock time for all four
# stages with no cross-process noise. Reused below for the zero-count total
# too, so this doesn't cost any extra invocations over the previous script.
backward_run_with_timing() {
  "$BINARY" "--spa-analysis=print-zeros=true time-passes=true pass-stage=backward" "$1" 2>&1 >/dev/null
}

time_for_stage() {
  echo "$1" | awk -v stage="$2" '$0 ~ ("  " stage "$") {print $1}'
}

printf "%-25s %10s %12s %10s %10s %10s %10s | %8s %8s %8s %8s %8s\n" \
  "Model" "Seed" "Fill+Pad" "Forward" "Lateral" "Backward" "Total" \
  "Seed(s)" "Fwd(s)" "Lat(s)" "Back(s)" "Total(s)"
printf "%-25s %10s %12s %10s %10s %10s %10s | %8s %8s %8s %8s %8s\n" \
  "-----" "----" "--------" "-------" "-------" "--------" "-----" \
  "-------" "------" "------" "-------" "--------"

for model in "$MLIR_DIR"/*.mlir; do
  name="$(basename "$model" .mlir)"

  after_seed=$(zeros_for_stage     "$model" seed)
  after_forward=$(zeros_for_stage  "$model" forward)
  after_lateral=$(zeros_for_stage  "$model" lateral)
  fill_pad=$(zeros_for_fill_and_pad "$model")

  backward_out=$(backward_run_with_timing "$model")
  after_backward=$(echo "$backward_out" | "$GREP" -oP 'Grand total: \K[0-9]+')

  delta_seed=$((     after_seed ))
  delta_forward=$((  after_forward  - after_seed ))
  delta_lateral=$((  after_lateral  - after_forward ))
  delta_backward=$(( after_backward - after_lateral ))

  time_seed=$(time_for_stage     "$backward_out" "Seed")
  time_forward=$(time_for_stage  "$backward_out" "Forward")
  time_lateral=$(time_for_stage  "$backward_out" "Lateral")
  time_backward=$(time_for_stage "$backward_out" "Backward")
  time_total=$(awk -v a="$time_seed" -v b="$time_forward" -v c="$time_lateral" -v d="$time_backward" \
    'BEGIN{printf "%.4f", a+b+c+d}')

  printf "%-25s %10s %12s %10s %10s %10s %10s | %8s %8s %8s %8s %8s\n" \
    "$name" "$delta_seed" "$fill_pad" "$delta_forward" "$delta_lateral" "$delta_backward" "$after_backward" \
    "$time_seed" "$time_forward" "$time_lateral" "$time_backward" "$time_total"
done
