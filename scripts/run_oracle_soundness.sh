#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MLIR_DIR="$ROOT_DIR/mlir_out"
export PROTEUS_BUILD_DIR="${PROTEUS_BUILD_DIR:-build-release}"

if [[ ! -d "$MLIR_DIR" ]]; then
  echo "Error: $MLIR_DIR not found. Run 'make dataset-convert' first." >&2
  exit 1
fi

passed=()
failed=()

for model in "$MLIR_DIR"/*.mlir; do
  name="$(basename "$model" .mlir)"
  echo "=== $name ==="
  if python3 "$SCRIPT_DIR/check_oracle_soundness.py" "$model"; then
    passed+=("$name")
  else
    failed+=("$name")
  fi
  echo
done

echo "==================== Summary ===================="
echo "Sound (${#passed[@]}): ${passed[*]:-none}"
echo "Violations (${#failed[@]}): ${failed[*]:-none}"

if [[ ${#failed[@]} -gt 0 ]]; then
  exit 1
fi
