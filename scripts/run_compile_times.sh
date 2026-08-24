#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MLIR_DIR="$ROOT_DIR/mlir_out_zerobias"

if ! command -v python3 &>/dev/null; then
  echo "Error: python3 is required." >&2
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
  "banded-16"
  "banded-32"
  "banded-64"
  "banded-128"
  "banded-192"
  "all-sparse"
)
LATTICE_ATTRS=(
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: -65536, -1, -1, 4294967295>}, {size = 224 : i64, words = array<i64: -65536, -1, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: -4294967296, -1, -1, 4294967295>}, {size = 224 : i64, words = array<i64: -4294967296, -1, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, -1, -1, 4294967295>}, {size = 224 : i64, words = array<i64: 0, -1, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, 0, -1, 4294967295>}, {size = 224 : i64, words = array<i64: 0, 0, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, 0, 0, 4294967295>}, {size = 224 : i64, words = array<i64: 0, 0, 0, 4294967295>}]"
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

compile_time_for_model() {
  local name="$1" attr="$2" rewrite="${3:-}"
  if [[ -n "$attr" && -n "$rewrite" ]]; then
    python3 "$SCRIPT_DIR/compile_times.py" --seed-lattice "$attr" --rewrite "$rewrite" \
      --warmup "$TIMING_WARMUP" --runs "$TIMING_RUNS" "$name" 2>/dev/null || true
  elif [[ -n "$attr" ]]; then
    python3 "$SCRIPT_DIR/compile_times.py" --seed-lattice "$attr" \
      --warmup "$TIMING_WARMUP" --runs "$TIMING_RUNS" "$name" 2>/dev/null || true
  else
    python3 "$SCRIPT_DIR/compile_times.py" \
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
  IFS=$'\t' read -r base_mean base_std base_samples <<< "$(compile_time_for_model "$name" "")"
  BASE_MEANS+=("$base_mean")
  BASE_STDS+=("$base_std")
done

if [[ "$CSV_OUTPUT" == "true" ]]; then
  echo "lattice,model,base_compile_mean,base_compile__std,compile_mean,compile_std"
fi

for li in "${!LATTICE_NAMES[@]}"; do
  lattice_name="${LATTICE_NAMES[$li]}"
  lattice_attr="${LATTICE_ATTRS[$li]}"

  if [[ "$CSV_OUTPUT" != "true" ]]; then
    echo "=== seed-lattice: $lattice_name ==="
    printf "%-25s | %17s %17s\n" "Model" "Base(s)" "Compile(s)"
    printf "%-25s | %17s %17s\n" "-----" "-----------------" "-----------------"
  fi

  model_index=0
  for model in "${MODELS[@]}"; do
    name="$(basename "$model" .mlir)"
    base_mean="${BASE_MEANS[$model_index]}"
    base_std="${BASE_STDS[$model_index]}"
    model_index=$((model_index + 1))

    IFS=$'\t' read -r compile_mean compile_std compile_samples <<< "$(compile_time_for_model "$name" "$lattice_attr" scf)"

    if [[ "$CSV_OUTPUT" == "true" ]]; then
      printf "%s,%s,%s,%s,%s,%s\n" \
        "$lattice_name" "$name" "$base_mean" "$base_std" "$compile_mean" "$compile_std"
    else
      base_str="$(fmt_mean_std "$base_mean" "$base_std")"
      compile_str="$(fmt_mean_std "$compile_mean" "$compile_std")"
      printf "%-25s | %17s %17s\n" "$name" "$base_str" "$compile_str"
    fi
  done
  if [[ "$CSV_OUTPUT" != "true" ]]; then
    echo
  fi
done
