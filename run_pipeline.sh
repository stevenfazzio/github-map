#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
LOG=data/pipeline.log
{
  echo "=== Pipeline started at $(date) ==="

  # Step 00: enumerate candidates via BigQuery (skip if already done)
  if [ ! -f data/candidates.csv ]; then
    echo "--- 00_enumerate_repos.py started at $(date) ---"
    python 00_enumerate_repos.py
    echo "--- 00_enumerate_repos.py finished at $(date) ---"
  else
    echo "--- Skipping 00_enumerate_repos.py (data/candidates.csv exists) ---"
  fi

  for script in 01_fetch_repos.py 01b_summarize_readmes.py 02_embed_readmes.py 03_reduce_umap.py 04_label_topics.py 05_visualize.py; do
    echo "--- $script started at $(date) ---"
    python "$script"
    echo "--- $script finished at $(date) ---"
  done
  echo "=== Pipeline finished at $(date) ==="
} 2>&1 | tee "$LOG"
