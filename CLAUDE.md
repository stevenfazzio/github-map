# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python data pipeline that analyzes the top 10,000 most-starred GitHub repositories. It enumerates candidates via BigQuery, fetches repo data via GraphQL direct lookups, embeds READMEs, reduces dimensionality, labels topics via LLM-driven topic modeling, and produces an interactive 2D visualization map (`github_map.html`).

## Running the Pipeline

Each stage is a standalone script run in order. There is no Makefile or test suite.

```bash
pip install -e .                      # Install dependencies
pip install -e '.[bigquery]'          # (optional) for 00_enumerate_repos.py

python 00_enumerate_repos.py          # BigQuery → data/candidates.csv (or use committed fallback)
python 01_fetch_repos.py              # GraphQL direct lookups → repos.parquet
python 01b_summarize_readmes.py       # Add LLM summaries, taglines, and target audience via Claude Haiku → updates repos.parquet
python 02_embed_readmes.py            # Embed READMEs via Cohere → embeddings.npz
python 03_reduce_umap.py              # UMAP 512-dim → 2D → umap_coords.npz
python 04_label_topics.py             # Toponymy + Claude Sonnet topic labels → labels.parquet
python 05_visualize.py                # DataMapPlot interactive HTML → github_map.html
```

**Two-phase fetch approach:** Step 00 queries GH Archive on BigQuery for repos with significant recent star activity, producing a generous candidate list (~25K). Step 01 then looks up each candidate via `repository(owner:, name:)` GraphQL queries (batched 50 per request), sorts by stars, and keeps the top 10K. This avoids the Search API's 1,000-result cap and non-deterministic ordering.

**Fallback for users without GCP:** A `candidates.csv` is committed to the repo root. If `data/candidates.csv` doesn't exist, `01_fetch_repos.py` copies from the committed file automatically. Most contributors never need BigQuery access.

`04b_experiment.py` is a side branch for comparing embedder configurations (MiniLM vs Cohere), outputting to `data/experiments/`.

## Required Environment Variables

Set in `.env` (loaded by `python-dotenv`):
- `GITHUB_TOKEN` — used by `01_fetch_repos.py`
- `CO_API_KEY` — Cohere API key, used by `02_embed_readmes.py`, `04_label_topics.py`, `04b_experiment.py`
- `ANTHROPIC_API_KEY` — used by `01b_summarize_readmes.py`, `04_label_topics.py`, `04b_experiment.py`
- `GCP_PROJECT` — (optional) Google Cloud project ID, used by `00_enumerate_repos.py`

## Architecture

**Sequential data pipeline** — each script reads outputs from previous stages:

```
candidates.csv ──> repos.parquet ──┬──> embeddings.npz ──> umap_coords.npz ──> labels.parquet
                                   │                                                │
                                   └────────────────────────────────────────────────┴──> github_map.html
```

**`config.py`** is the central configuration hub: all file paths, API keys, and constants (batch sizes, model names, target counts) are defined here. Every pipeline script imports from it.

**Key technology choices:**
- BigQuery (GH Archive) for reliable repo enumeration
- GraphQL `repository()` direct lookups for metadata + READMEs (batched 25/request)
- Cohere `embed-v4.0` (512-dim) for README embeddings
- UMAP (n_neighbors=15, min_dist=0.05, cosine) for dimensionality reduction (512D → 2D)
- Toponymy library for hierarchical topic clustering with LLM-generated labels
- DataMapPlot for the final interactive HTML visualization
- Claude Haiku for README summarization, Claude Sonnet for topic naming

## Data Pipeline Rules

- NEVER overwrite existing parquet/data files in-place. Always write to new files (e.g., with timestamp or version suffix) and only replace originals after verification. Treat fetched data as expensive/irreplaceable.
- For GitHub API calls (GraphQL or REST), always implement retry with exponential backoff, handling 502, ReadTimeout, and ChunkedEncodingError. Never assume a single request pattern will scale.

## Data Directory

All outputs go to `data/` (gitignored). Key files: `candidates.csv`, `repos.parquet`, `embeddings.npz`, `umap_coords.npz`, `labels.parquet`, `toponymy_model.joblib`, `github_map.html`.

## Visualization Details

`05_visualize.py` produces the main output — an interactive HTML map with multiple colormaps (language, stars, license, age), hover tooltips with summaries, click-to-open-repo, and search. Toponymy's hierarchical cluster layers are passed directly to DataMapPlot for multi-level topic label display.

## docs/ Directory: Source vs. Generated Files

- `docs/methodology.html` — **hand-authored source file**. Edit it directly for methodology content changes.
- `docs/index.html` — **generated output** from `05_visualize.py` (copy of `data/github_map.html` with adjusted links). Do not edit directly.
- `data/methodology.html` — **generated copy** of `docs/methodology.html` with nav links adjusted (`index.html` → `github_map.html`) for local use.
