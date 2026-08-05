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

CSV_OUTPUT=false
for arg in "$@"; do
  case "$arg" in
    --csv) CSV_OUTPUT=true ;;
    *)
      echo "Error: unknown argument '$arg'" >&2
      exit 1
      ;;
  esac
done

TIMING_WARMUP="${TIMING_WARMUP:-3}"
TIMING_RUNS="${TIMING_RUNS:-10}"

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
  LATTICE_NAMES=()
  LATTICE_ATTRS=()
  [[ ${#FILTERED_NAMES[@]} -gt 0 ]] && LATTICE_NAMES=("${FILTERED_NAMES[@]}")
  [[ ${#FILTERED_ATTRS[@]} -gt 0 ]] && LATTICE_ATTRS=("${FILTERED_ATTRS[@]}")
fi

if [[ ${#LATTICE_NAMES[@]} -eq 0 ]]; then
  echo "Error: no lattices to check (LATTICE_FILTER=${LATTICE_FILTER:-unset} matched none)." >&2
  exit 1
fi

seed_opt() {
  if [[ -n "$1" ]]; then
    printf " seed-lattice='%s'" "$1"
  fi
}

time_for_stage() {
  echo "$1" | awk -v stage="$2" '$0 ~ ("  " stage "$") {print $1}'
}

mean_std() {
  if [[ $# -eq 0 ]]; then
    printf "n/a\tn/a"
    return
  fi
  printf '%s\n' "$@" | awk '{a[NR-1]=$1} END{
    n=NR; s=0; for(i=0;i<n;i++) s+=a[i]; m=s/n;
    if (n>1) { ss=0; for(i=0;i<n;i++) ss+=(a[i]-m)^2; sd=sqrt(ss/(n-1)) } else { sd=0 }
    printf "%.4f\t%.4f", m, sd
  }'
}

analysis_timings_for_model() {
  local model="$1" attr="$2"
  local seeds=() fwds=() lats=() backs=() totals=()
  local i out ts tf tl tb tt
  for ((i = 0; i < TIMING_WARMUP + TIMING_RUNS; i++)); do
    out="$("$BINARY" "--spa-analysis=time-passes=true pass-stage=backward$(seed_opt "$attr")" "$model" -o /dev/null 2>&1)"
    if ((i >= TIMING_WARMUP)); then
      ts=$(time_for_stage "$out" "Seed")
      tf=$(time_for_stage "$out" "Forward")
      tl=$(time_for_stage "$out" "Lateral")
      tb=$(time_for_stage "$out" "Backward")
      tt=$(awk -v a="$ts" -v b="$tf" -v c="$tl" -v d="$tb" 'BEGIN{printf "%.4f", a+b+c+d}')
      seeds+=("$ts")
      fwds+=("$tf")
      lats+=("$tl")
      backs+=("$tb")
      totals+=("$tt")
    fi
  done
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$(mean_std "${seeds[@]}")" "$(mean_std "${fwds[@]}")" \
    "$(mean_std "${lats[@]}")" "$(mean_std "${backs[@]}")" "$(mean_std "${totals[@]}")"
}

rewrite_pass_time_for_model() {
  local model="$1" attr="$2" target="$3"
  local times=()
  local i out t
  for ((i = 0; i < TIMING_WARMUP + TIMING_RUNS; i++)); do
    out="$("$BINARY" "--spa-analysis=lattice-dump=true$(seed_opt "$attr")" \
      "--spa-rewrite=target=$target time-rewrite=true" "$model" -o /dev/null 2>&1)"
    if ((i >= TIMING_WARMUP)); then
      times+=("$(time_for_stage "$out" "Rewrite")")
    fi
  done
  mean_std "${times[@]}"
}

timed_runs_for_model() {
  local name="$1" attr="$2" rewrite="${3:-}"
  if [[ -n "$rewrite" ]]; then
    python3 "$SCRIPT_DIR/benchmark_runtime.py" --seed-lattice "$attr" --rewrite "$rewrite" \
      --warmup "$TIMING_WARMUP" --runs "$TIMING_RUNS" "$name" 2>/dev/null || true
  else
    python3 "$SCRIPT_DIR/benchmark_runtime.py" --seed-lattice "$attr" \
      --warmup "$TIMING_WARMUP" --runs "$TIMING_RUNS" "$name" 2>/dev/null || true
  fi
}

fmt_mean_std() {
  local mean="$1" std="$2"
  if [[ -z "$mean" || "$mean" == "n/a" ]]; then
    echo "n/a"
    return
  fi
  awk -v m="$mean" -v s="$std" 'BEGIN{printf "%.4f\xc2\xb1%.4f", m, s}'
}

speedup_for_means() {
  local base="$1" val="$2"
  if [[ -z "$base" || "$base" == "n/a" || -z "$val" || "$val" == "n/a" ]]; then
    echo "n/a"
    return
  fi
  awk -v b="$base" -v r="$val" 'BEGIN{ if (r > 0) printf "%.2fx", b/r; else print "n/a" }'
}

MODELS=("$MLIR_DIR"/*.mlir)

if [[ -n "${MODEL_FILTER:-}" ]]; then
  FILTERED_MODELS=()
  for model in "${MODELS[@]}"; do
    name="$(basename "$model" .mlir)"
    if [[ ",${MODEL_FILTER}," == *",${name},"* ]]; then
      FILTERED_MODELS+=("$model")
    fi
  done
  MODELS=()
  [[ ${#FILTERED_MODELS[@]} -gt 0 ]] && MODELS=("${FILTERED_MODELS[@]}")
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "Error: no models to check (MODEL_FILTER=${MODEL_FILTER:-unset} matched none in $MLIR_DIR)." >&2
  exit 1
fi

BASE_MEANS=()
BASE_STDS=()
for model in "${MODELS[@]}"; do
  name="$(basename "$model" .mlir)"
  IFS=$'\t' read -r base_mean base_std <<< "$(timed_runs_for_model "$name" "")"
  BASE_MEANS+=("$base_mean")
  BASE_STDS+=("$base_std")
done

if [[ "$CSV_OUTPUT" == "true" ]]; then
  echo "lattice,model,seed_mean,seed_std,fwd_mean,fwd_std,lat_mean,lat_std,back_mean,back_std,total_mean,total_std,rwscf_mean,rwscf_std,rwscf_speedup,speedup,base_mean,base_std,scf_mean,scf_std,scf_speedup"
fi

for li in "${!LATTICE_NAMES[@]}"; do
  lattice_name="${LATTICE_NAMES[$li]}"
  lattice_attr="${LATTICE_ATTRS[$li]}"

  if [[ "$CSV_OUTPUT" != "true" ]]; then
    echo "=== seed-lattice: $lattice_name ==="
    printf "%-25s | %17s %17s %17s %17s %17s | %17s %12s | %8s | %17s %17s | %11s\n" \
      "Model" "Seed(s)" "Fwd(s)" "Lat(s)" "Back(s)" "Total(s)" \
      "RwScf(s)" "Speedup (RW)" \
      "Speedup" "Base(s)" "Scf(s)" "Speedup (S)"
    printf "%-25s | %17s %17s %17s %17s %17s | %17s %12s | %8s | %17s %17s | %11s\n" \
      "-----" "-----------------" "-----------------" "-----------------" "-----------------" "-----------------" \
      "-----------------" "------------" \
      "-------" "-----------------" "-----------------" "-----------"
  fi

  model_index=0
  for model in "${MODELS[@]}"; do
    name="$(basename "$model" .mlir)"
    base_mean="${BASE_MEANS[$model_index]}"
    base_std="${BASE_STDS[$model_index]}"
    model_index=$((model_index + 1))

    IFS=$'\t' read -r seed_mean seed_std fwd_mean fwd_std lat_mean lat_std back_mean back_std total_mean total_std \
      <<< "$(analysis_timings_for_model "$model" "$lattice_attr")"

    IFS=$'\t' read -r rwscf_mean rwscf_std <<< "$(rewrite_pass_time_for_model "$model" "$lattice_attr" scf)"

    IFS=$'\t' read -r scf_mean scf_std <<< "$(timed_runs_for_model "$name" "$lattice_attr" scf)"

    seed_str="$(fmt_mean_std "$seed_mean" "$seed_std")"
    fwd_str="$(fmt_mean_std "$fwd_mean" "$fwd_std")"
    lat_str="$(fmt_mean_std "$lat_mean" "$lat_std")"
    back_str="$(fmt_mean_std "$back_mean" "$back_std")"
    total_str="$(fmt_mean_std "$total_mean" "$total_std")"
    rwscf_str="$(fmt_mean_std "$rwscf_mean" "$rwscf_std")"
    base_str="$(fmt_mean_std "$base_mean" "$base_std")"
    scf_str="$(fmt_mean_std "$scf_mean" "$scf_std")"

    speedup="$(speedup_for_means "$base_mean" "$total_mean")"
    rwscf_speedup="$(speedup_for_means "$base_mean" "$rwscf_mean")"
    scf_speedup="$(speedup_for_means "$base_mean" "$scf_mean")"

    if [[ "$CSV_OUTPUT" == "true" ]]; then
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$lattice_name" "$name" \
        "$seed_mean" "$seed_std" "$fwd_mean" "$fwd_std" "$lat_mean" "$lat_std" "$back_mean" "$back_std" "$total_mean" "$total_std" \
        "$rwscf_mean" "$rwscf_std" "${rwscf_speedup%x}" \
        "${speedup%x}" "$base_mean" "$base_std" "$scf_mean" "$scf_std" "${scf_speedup%x}"
    else
      printf "%-25s | %18s %18s %18s %18s %18s | %18s %12s | %8s | %18s %18s | %11s\n" \
        "$name" "$seed_str" "$fwd_str" "$lat_str" "$back_str" "$total_str" \
        "$rwscf_str" "$rwscf_speedup" \
        "$speedup" "$base_str" "$scf_str" "$scf_speedup"
    fi
  done
  if [[ "$CSV_OUTPUT" != "true" ]]; then
    echo
  fi
done
