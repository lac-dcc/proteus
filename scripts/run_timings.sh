#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PROTEUS_BUILD_DIR="${PROTEUS_BUILD_DIR:-build-release}"
BINARY="$ROOT_DIR/$PROTEUS_BUILD_DIR/bin/proteus-opt"
MLIR_DIR="$ROOT_DIR/mlir_out_zerobias"

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
  echo "Error: python3 is required (used for the runtime/rewrite-speedup columns)." >&2
  exit 1
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
)

if [[ -n "${LATTICE_FILTER:-}" ]]; then
  FILTERED_NAMES=()
  FILTERED_ATTRS=()
  for i in "${!LATTICE_NAMES[@]}"; do
    if [[ ",${LATTICE_FILTER}," == *",${LATTICE_NAMES[$i]},"* ]]; then
      FILTERED_NAMES+=("${LATTICE_NAMES[$i]}")
      FILTERED_ATTRS+=("${LATTICE_ATTRS[$i]}")
    fi
  done
  LATTICE_NAMES=("${FILTERED_NAMES[@]}")
  LATTICE_ATTRS=("${FILTERED_ATTRS[@]}")
fi

seed_opt() {
  if [[ -n "$1" ]]; then
    printf " seed-lattice='%s'" "$1"
  fi
}

backward_run_with_timing() {
  local model="$1" attr="$2"
  "$BINARY" "--spa-analysis=print-zeros=true time-passes=true pass-stage=backward$(seed_opt "$attr")" "$model" 2>&1 >/dev/null
}

time_for_stage() {
  echo "$1" | awk -v stage="$2" '$0 ~ ("  " stage "$") {print $1}'
}

# Wall time (seconds) the --spa-rewrite pass itself took to apply its
# patterns (compile-time cost of the rewrite, not the rewritten model's
# runtime), via --spa-rewrite=...time-rewrite=true.
rewrite_pass_time_for_model() {
  local model="$1" attr="$2" target="$3"
  local out t
  out="$("$BINARY" "--spa-analysis=lattice-dump=true$(seed_opt "$attr")" \
    "--spa-rewrite=target=$target time-rewrite=true" "$model" 2>&1 >/dev/null || true)"
  t="$(time_for_stage "$out" "Rewrite")"
  [[ -z "$t" ]] && t="n/a"
  echo "$t"
}

timed_runs_for_model() {
  local name="$1" attr="$2" rewrite="${3:-}"
  if [[ -n "$rewrite" ]]; then
    python3 "$SCRIPT_DIR/benchmark_runtime.py" --seed-lattice "$attr" --rewrite "$rewrite" "$name" 2>/dev/null || true
  else
    python3 "$SCRIPT_DIR/benchmark_runtime.py" --seed-lattice "$attr" "$name" 2>/dev/null || true
  fi
}

fmt_mean_std() {
  local mean="$1" std="$2"
  if [[ -z "$mean" ]]; then
    echo "n/a"
    return
  fi
  awk -v m="$mean" -v s="$std" 'BEGIN{printf "%.4f\xc2\xb1%.4f", m, s}'
}

speedup_for_means() {
  local base="$1" val="$2"
  if [[ -z "$base" || -z "$val" ]]; then
    echo "n/a"
    return
  fi
  awk -v b="$base" -v r="$val" 'BEGIN{ if (r > 0) printf "%.2fx", b/r; else print "n/a" }'
}

MODELS=("$MLIR_DIR"/*.mlir)

for li in "${!LATTICE_NAMES[@]}"; do
  lattice_name="${LATTICE_NAMES[$li]}"
  lattice_attr="${LATTICE_ATTRS[$li]}"

  echo "=== seed-lattice: $lattice_name ==="
  printf "%-25s | %8s %8s %8s %8s %8s | %9s %9s | %8s | %17s %17s %17s | %11s %11s\n" \
    "Model" "Seed(s)" "Fwd(s)" "Lat(s)" "Back(s)" "Total(s)" \
    "RwLin(s)" "RwScf(s)" \
    "Speedup" "Base(s)" "Linalg(s)" "Scf(s)" "Speedup (L)" "Speedup (S)"
  printf "%-25s | %8s %8s %8s %8s %8s | %9s %9s | %8s | %17s %17s %17s | %11s %11s\n" \
    "-----" "-------" "------" "------" "-------" "--------" \
    "--------" "--------" \
    "-------" "-----------------" "-----------------" "-----------------" "-----------" "-----------"

  for model in "${MODELS[@]}"; do
    name="$(basename "$model" .mlir)"

    backward_out=$(backward_run_with_timing "$model" "$lattice_attr")
    time_seed=$(time_for_stage     "$backward_out" "Seed")
    time_forward=$(time_for_stage  "$backward_out" "Forward")
    time_lateral=$(time_for_stage  "$backward_out" "Lateral")
    time_backward=$(time_for_stage "$backward_out" "Backward")
    time_total=$(awk -v a="$time_seed" -v b="$time_forward" -v c="$time_lateral" -v d="$time_backward" \
      'BEGIN{printf "%.4f", a+b+c+d}')

    rw_linalg_time="$(rewrite_pass_time_for_model "$model" "$lattice_attr" linalg)"
    rw_scf_time="$(rewrite_pass_time_for_model "$model" "$lattice_attr" scf)"

    IFS=$'\t' read -r base_mean base_std <<< "$(timed_runs_for_model "$name" "$lattice_attr")"
    IFS=$'\t' read -r linalg_mean linalg_std <<< "$(timed_runs_for_model "$name" "$lattice_attr" linalg)"
    IFS=$'\t' read -r scf_mean scf_std <<< "$(timed_runs_for_model "$name" "$lattice_attr" scf)"

    base_str="$(fmt_mean_std "$base_mean" "$base_std")"
    linalg_str="$(fmt_mean_std "$linalg_mean" "$linalg_std")"
    scf_str="$(fmt_mean_std "$scf_mean" "$scf_std")"
    speedup="$(speedup_for_means "$base_mean" "$time_total")"
    linalg_speedup="$(speedup_for_means "$base_mean" "$linalg_mean")"
    scf_speedup="$(speedup_for_means "$base_mean" "$scf_mean")"

    printf "%-25s | %8s %8s %8s %8s %8s | %9s %9s | %8s | %18s %18s %18s | %11s %11s\n" \
      "$name" "$time_seed" "$time_forward" "$time_lateral" "$time_backward" "$time_total" \
      "$rw_linalg_time" "$rw_scf_time" \
      "$speedup" "$base_str" "$linalg_str" "$scf_str" "$linalg_speedup" "$scf_speedup"
  done
  echo
done
