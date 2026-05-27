#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# MAPS=("simple_spread" "simple_world")
MAPS=("simple_world")
# QUALITIES=("expert" "medium" "medium-replay" "random")
QUALITIES=("medium" "medium-replay" "random")
WORKERS="${1:-10}"  # override via: ./run_mpe_transform_all.sh 8

echo "Using ${WORKERS} parallel worker processes."
echo

echo "[1/2] Running transform_omar_dataset.py for all map/quality combinations..."
for map_name in "${MAPS[@]}"; do
  for quality in "${QUALITIES[@]}"; do
    echo "--- python scripts/transform_omar_dataset.py --env_name mpe --map_name ${map_name} --quality ${quality} --workers ${WORKERS}"
    python scripts/transform_omar_dataset.py \
        --env_name mpe \
        --map_name "${map_name}" \
        --quality "${quality}" \
        --workers "${WORKERS}"
  done
done

echo "[2/2] Running transform_og_marl_dataset.py for all map/quality combinations..."
for map_name in "${MAPS[@]}"; do
  for quality in "${QUALITIES[@]}"; do
    echo "python scripts/transform_og_marl_dataset.py --env_name mpe --map_name ${map_name} --quality ${quality}"
    python scripts/transform_og_marl_dataset.py --env_name mpe --map_name "${map_name}" --quality "${quality}"
  done
done

echo "All commands finished."
