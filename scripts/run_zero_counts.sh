#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PROTEUS_BUILD_DIR="${PROTEUS_BUILD_DIR:-build-release}"
BINARY="$ROOT_DIR/$PROTEUS_BUILD_DIR/bin/proteus-opt"
MLIR_DIR="$ROOT_DIR/mlir_out"
RUNTIME_CACHE="$SCRIPT_DIR/.model_runtimes.tsv"

if [[ ! -x "$BINARY" ]]; then
  echo "Error: $BINARY not found or not executable. Run 'make $PROTEUS_BUILD_DIR' first (or set PROTEUS_BUILD_DIR=build to use a debug build)." >&2
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

if ! command -v python3 &>/dev/null; then
  echo "Error: python3 is required (used for the breakoff-percentage columns)." >&2
  exit 1
fi

if [[ ! -f "$RUNTIME_CACHE" ]]; then
  echo "No runtime cache found; benchmarking single-iteration model runtimes first..." >&2
  python3 "$SCRIPT_DIR/benchmark_runtime.py"
fi

LATTICE_NAMES=(
  "banded-64"
  "banded-128"
  "banded-192"
  "banded-64-strided"
  "all-dense"
  "all-sparse"
)
LATTICE_ATTRS=(
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, -1, -1, 4294967295>}, {size = 224 : i64, words = array<i64: 0, -1, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, 0, -1, 4294967295>}, {size = 224 : i64, words = array<i64: 0, 0, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, 0, 0, 4294967295>}, {size = 224 : i64, words = array<i64: 0, 0, 0, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, -1, 0, 4294967295>}, {size = 224 : i64, words = array<i64: 0, -1, 0, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: -1, -1, -1, 4294967295>}, {size = 224 : i64, words = array<i64: -1, -1, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, 0, 0, 0>}, {size = 224 : i64, words = array<i64: 0, 0, 0, 0>}]"
  # "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 9150747060186627966, -4647998506761461825, -2323999253380730913, 4261148655>}, {size = 224 : i64, words = array<i64: 9150747060186627966, -4647998506761461825, -2323999253380730913, 4261148655>}]"
  # "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 9005497106850332284, 4502748553425166142, -6971997760142192737, 4193511375>}, {size = 224 : i64, words = array<i64: 9005497106850332284, 4502748553425166142, -6971997760142192737, 4193511375>}]"
  # "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 8133997386832558192, 4066998693416279096, 2033499346708139548, 3787687694>}, {size = 224 : i64, words = array<i64: 8133997386832558192, 4066998693416279096, 2033499346708139548, 3787687694>}]"
)

seed_opt() {
  if [[ -n "$1" ]]; then
    printf " seed-lattice='%s'" "$1"
  fi
}

zeros_for_stage() {
  local model="$1" stage="$2" attr="$3"
  "$BINARY" "--spa-analysis=print-zeros=true pass-stage=$stage$(seed_opt "$attr")" "$model" 2>&1 >/dev/null \
    | "$GREP" -oP 'Grand total: \K[0-9]+'
}

zeros_for_fill_and_pad() {
  local model="$1" attr="$2"
  "$BINARY" "--spa-analysis=print-zeros=true pass-stage=forward$(seed_opt "$attr")" "$model" 2>&1 >/dev/null \
    | "$GREP" -A1 -E '\[(linalg\.fill|tensor\.pad)\]' \
    | "$GREP" -oP 'total: \K[0-9]+' \
    | awk '{s+=$1} END{print s+0}'
}

backward_run_with_timing() {
  local model="$1" attr="$2"
  "$BINARY" "--spa-analysis=print-zeros=true time-passes=true pass-stage=backward$(seed_opt "$attr")" "$model" 2>&1 >/dev/null
}

time_for_stage() {
  echo "$1" | awk -v stage="$2" '$0 ~ ("  " stage "$") {print $1}'
}

breakoff_for_model() {
  local model="$1" attr="$2"
  python3 "$SCRIPT_DIR/find_sparsity_breakoff.py" "$model" "$attr"
}

runtime_for_model() {
  local name="$1"
  awk -F'\t' -v name="$name" '$1 == name {printf "%.4f", $2}' "$RUNTIME_CACHE"
}

for li in "${!LATTICE_NAMES[@]}"; do
  lattice_name="${LATTICE_NAMES[$li]}"
  lattice_attr="${LATTICE_ATTRS[$li]}"

  echo "=== seed-lattice: $lattice_name ==="
  printf "%-25s %10s %12s %10s %10s %10s %10s | %8s %8s %8s %8s %8s | %9s %9s | %10s %12s\n" \
    "Model" "Seed" "Fill+Pad" "Forward" "Lateral" "Backward" "Total" \
    "Seed(s)" "Fwd(s)" "Lat(s)" "Back(s)" "Total(s)" "Breakoff" "1st MM" "Run 1x(s)" "Speedup"
  printf "%-25s %10s %12s %10s %10s %10s %10s | %8s %8s %8s %8s %8s | %9s %9s | %10s %12s\n" \
    "-----" "----" "--------" "-------" "-------" "--------" "-----" \
    "-------" "------" "------" "-------" "--------" "--------" "------" "---------" "-------"

  for model in "$MLIR_DIR"/*.mlir; do
    name="$(basename "$model" .mlir)"

    after_seed=$(zeros_for_stage     "$model" seed    "$lattice_attr")
    after_forward=$(zeros_for_stage  "$model" forward "$lattice_attr")
    after_lateral=$(zeros_for_stage  "$model" lateral "$lattice_attr")
    fill_pad=$(zeros_for_fill_and_pad "$model" "$lattice_attr")

    backward_out=$(backward_run_with_timing "$model" "$lattice_attr")
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

    IFS=',' read -r breakoff_pct matmul_pct <<< "$(breakoff_for_model "$model" "$lattice_attr")"
    breakoff_str="${breakoff_pct}%"
    matmul_str="${matmul_pct}"
    [[ "$matmul_str" != "None" ]] && matmul_str="${matmul_str}%"

    run_once="$(runtime_for_model "$name")"
    [[ -z "$run_once" ]] && run_once="n/a"

    speedup="None"
    if [[ "$run_once" != "None" ]]; then
      speedup=$(awk -v r="$run_once" -v t="$time_total" \
        'BEGIN{ if (t > 0) printf "%.1fx", r/t; else print "None" }')
    fi

    printf "%-25s %10s %12s %10s %10s %10s %10s | %8s %8s %8s %8s %8s | %9s %9s | %10s %12s\n" \
      "$name" "$delta_seed" "$fill_pad" "$delta_forward" "$delta_lateral" "$delta_backward" "$after_backward" \
      "$time_seed" "$time_forward" "$time_lateral" "$time_backward" "$time_total" \
      "$breakoff_str" "$matmul_str" "$run_once" "$speedup"
  done
  echo
done
