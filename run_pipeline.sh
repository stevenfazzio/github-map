#!/usr/bin/env bash
set -euo pipefail
LOG=data/pipeline.log
{
  echo "=== Pipeline started at $(date) ==="
  for script in 01_fetch_repos.py 01b_summarize_readmes.py 02_embed_readmes.py 03_reduce_umap.py 04_label_topics.py 05_visualize.py; do
    echo "--- $script started at $(date) ---"
    python "$script"
    echo "--- $script finished at $(date) ---"
  done
  echo "=== Pipeline finished at $(date) ==="
} 2>&1 | tee "$LOG"
