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
  echo "Error: python3 is required." >&2
  exit 1
fi

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

oracle_for_model() {
  local model="$1" attr="$2"
  if python3 "$SCRIPT_DIR/check_oracle_soundness.py" "$model" "$attr" >/dev/null 2>&1; then
    echo "PASS"
  else
    echo "FAIL"
  fi
}

rewrite_correctness_for_model() {
  local model="$1" attr="$2"
  local output scf_result
  output="$(python3 "$SCRIPT_DIR/check_rewrite_correctness.py" "$model" "$attr" 2>/dev/null || true)"
  scf_result="$(echo "$output" | "$GREP" -oP '^scf: \K\S+' || true)"
  [[ -z "$scf_result" ]] && scf_result="FAIL"
  echo "$scf_result"
}

if [[ ! -d "$MLIR_DIR" ]]; then
  echo "Error: $MLIR_DIR not found. Run 'make dataset-convert' first." >&2
  exit 1
fi

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

overall_pass=true
for li in "${!LATTICE_NAMES[@]}"; do
  lattice_name="${LATTICE_NAMES[$li]}"
  lattice_attr="${LATTICE_ATTRS[$li]}"

  echo "=== seed-lattice: $lattice_name ==="
  printf "%-25s | %10s | %10s\n" "Model" "Analysis" "Rewrite (S)"
  printf "%-25s | %10s | %10s\n" "-----" "--------" "-----------"

  for model in "${MODELS[@]}"; do
    name="$(basename "$model" .mlir)"

    oracle_result="$(oracle_for_model "$model" "$lattice_attr")"
    scf_result="$(rewrite_correctness_for_model "$model" "$lattice_attr")"

    printf "%-25s | %10s | %10s\n" "$name" "$oracle_result" "$scf_result"

    if [[ "$oracle_result" != "PASS" || "$scf_result" != "PASS" ]]; then
      overall_pass=false
    fi
  done
  echo
done

if [[ "$overall_pass" != "true" ]]; then
  exit 1
fi
