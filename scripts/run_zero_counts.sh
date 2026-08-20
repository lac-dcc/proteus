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
  echo "Error: python3 is required (used for the breakoff-percentage columns)." >&2
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

LATTICE_NAMES=(
  "banded-16"
  "banded-32"
  "banded-64"
  "banded-128"
  "banded-192"
  "banded-16-strided"
  "banded-32-strided"
  "banded-64-strided"
  "all-sparse"
)
LATTICE_ATTRS=(
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: -65536, -1, -1, 4294967295>}, {size = 224 : i64, words = array<i64: -65536, -1, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: -4294967296, -1, -1, 4294967295>}, {size = 224 : i64, words = array<i64: -4294967296, -1, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, -1, -1, 4294967295>}, {size = 224 : i64, words = array<i64: 0, -1, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, 0, -1, 4294967295>}, {size = 224 : i64, words = array<i64: 0, 0, -1, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, 0, 0, 4294967295>}, {size = 224 : i64, words = array<i64: 0, 0, 0, 4294967295>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: -281470681808896, -281470681808896, -281470681808896, 4294901760>}, {size = 224 : i64, words = array<i64: -281470681808896, -281470681808896, -281470681808896, 4294901760>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: -4294967296, -4294967296, -4294967296, 0>}, {size = 224 : i64, words = array<i64: -4294967296, -4294967296, -4294967296, 0>}]"
  "[{size = 1 : i64, words = array<i64: 1>}, {size = 3 : i64, words = array<i64: 7>}, {size = 224 : i64, words = array<i64: 0, -1, 0, 4294967295>}, {size = 224 : i64, words = array<i64: 0, -1, 0, 4294967295>}]"
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

zeros_for_stage() {
  local model="$1" stage="$2" attr="$3"
  "$BINARY" "--spa-analysis=print-zeros=true pass-stage=$stage$(seed_opt "$attr")" "$model" 2>&1 >/dev/null \
    | "$GREP" -oP 'Grand total: \K[0-9]+'
}

run_forward_stage() {
  local model="$1" attr="$2"
  "$BINARY" "--spa-analysis=print-zeros=true pass-stage=forward$(seed_opt "$attr")" "$model" 2>&1 >/dev/null
}

fill_pad_broadcast_const_from_output() {
  { "$GREP" -A1 -E '\[(linalg\.fill|tensor\.pad|linalg\.broadcast|arith\.constant)\]' || true; } \
    | { "$GREP" -oP 'total: \K[0-9]+' || true; } \
    | awk '{s+=$1} END{print s+0}'
}

breakoff_for_model() {
  local model="$1" attr="$2"
  python3 "$SCRIPT_DIR/find_sparsity_breakoff.py" "$model" "$attr"
}

matmul_for_model() {
  local model="$1"
  python3 "$SCRIPT_DIR/find_first_matmul.py" "$model"
}

rewrite_counts_for_model() {
  local model="$1" attr="$2"
  local out matmul_rw conv_rw depth_rw pool_rw poolmax_rw
  out="$("$BINARY" "--spa-analysis=lattice-dump=true$(seed_opt "$attr")" \
    "--spa-rewrite=count-rewrites=true target=scf" "$model" 2>&1 >/dev/null || true)"
  matmul_rw="$(echo "$out" | "$GREP" -oP 'linalg\.matmul=\K[0-9]+' | awk '{s+=$1} END{print s+0}')"
  conv_rw="$(echo "$out" | "$GREP" -oP 'linalg\.conv_2d_nchw_fchw=\K[0-9]+' | awk '{s+=$1} END{print s+0}')"
  depth_rw="$(echo "$out" | "$GREP" -oP 'linalg\.depthwise_conv_2d_nchw_chw=\K[0-9]+' | awk '{s+=$1} END{print s+0}')"
  pool_rw="$(echo "$out" | "$GREP" -oP 'linalg\.pooling_nchw_sum=\K[0-9]+' | awk '{s+=$1} END{print s+0}')"
  poolmax_rw="$(echo "$out" | "$GREP" -oP 'linalg\.pooling_nchw_max=\K[0-9]+' | awk '{s+=$1} END{print s+0}')"
  echo "${matmul_rw}|${conv_rw}|${depth_rw}|${pool_rw}|${poolmax_rw}"
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

MATMUL_PCTS=()
for model in "${MODELS[@]}"; do
  MATMUL_PCTS+=("$(matmul_for_model "$model")")
done

if [[ "$CSV_OUTPUT" == "true" ]]; then
  echo "lattice,model,seed,fill_pad_broadcast_const,forward,lateral,backward,total,breakoff_pct,first_matmul_pct,matmul_rw,conv_rw,depthwise_rw,poolsum_rw,poolmax_rw"
fi

for li in "${!LATTICE_NAMES[@]}"; do
  lattice_name="${LATTICE_NAMES[$li]}"
  lattice_attr="${LATTICE_ATTRS[$li]}"

  if [[ "$CSV_OUTPUT" != "true" ]]; then
    echo "=== seed-lattice: $lattice_name ==="
    printf "%-25s %10s %12s %10s %10s %10s %10s | %9s %9s | %6s %6s %6s %7s %7s\n" \
      "Model" "Seed" "Fill+Pad+Bc+C" "Forward" "Lateral" "Backward" "Total" \
      "Breakoff" "1st MM" "MM RW" "Cnv RW" "DW RW" "PSum RW" "PMax RW"
    printf "%-25s %10s %12s %10s %10s %10s %10s | %9s %9s | %6s %6s %6s %7s %7s\n" \
      "-----" "----" "--------" "-------" "-------" "--------" "-----" \
      "--------" "------" "------" "------" "------" "-------" "-------"
  fi

  model_index=0
  for model in "${MODELS[@]}"; do
    name="$(basename "$model" .mlir)"
    matmul_pct="${MATMUL_PCTS[$model_index]}"
    model_index=$((model_index + 1))

    after_seed=$(zeros_for_stage    "$model" seed    "$lattice_attr")
    forward_out=$(run_forward_stage "$model" "$lattice_attr")
    after_forward=$(echo "$forward_out" | "$GREP" -oP 'Grand total: \K[0-9]+')
    fill_pad_broadcast_const=$(echo "$forward_out" | fill_pad_broadcast_const_from_output)
    after_lateral=$(zeros_for_stage "$model" lateral  "$lattice_attr")
    after_backward=$(zeros_for_stage "$model" backward "$lattice_attr")

    delta_seed=$((     after_seed ))
    delta_forward=$((  after_forward  - after_seed ))
    delta_lateral=$((  after_lateral  - after_forward ))
    delta_backward=$(( after_backward - after_lateral ))

    breakoff_pct="$(breakoff_for_model "$model" "$lattice_attr")"
    breakoff_str="${breakoff_pct}%"
    matmul_str="${matmul_pct}"
    [[ "$matmul_str" != "None" ]] && matmul_str="${matmul_str}%"

    IFS='|' read -r matmul_rw conv_rw depth_rw pool_rw poolmax_rw <<< "$(rewrite_counts_for_model "$model" "$lattice_attr")"

    if [[ "$CSV_OUTPUT" == "true" ]]; then
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$lattice_name" "$name" "$delta_seed" "$fill_pad_broadcast_const" "$delta_forward" "$delta_lateral" "$delta_backward" "$after_backward" \
        "$breakoff_pct" "$matmul_pct" "$matmul_rw" "$conv_rw" "$depth_rw" "$pool_rw" "$poolmax_rw"
    else
      printf "%-25s %10s %12s %10s %10s %10s %10s | %9s %9s | %6s %6s %6s %7s %7s\n" \
        "$name" "$delta_seed" "$fill_pad_broadcast_const" "$delta_forward" "$delta_lateral" "$delta_backward" "$after_backward" \
        "$breakoff_str" "$matmul_str" "$matmul_rw" "$conv_rw" "$depth_rw" "$pool_rw" "$poolmax_rw"
    fi
  done
  if [[ "$CSV_OUTPUT" != "true" ]]; then
    echo
  fi
done
