# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python data pipeline that analyzes the top 1,000 most-starred GitHub repositories. It fetches repo data, embeds READMEs, reduces dimensionality, labels topics via LLM-driven topic modeling, and produces an interactive 2D visualization map (`github_map.html`).

## Running the Pipeline

Each stage is a standalone script run in order. There is no Makefile or test suite.

```bash
pip install -e .                      # Install dependencies

python 01_fetch_repos.py              # Fetch top 1K repos from GitHub API → repos.parquet
python 01b_summarize_readmes.py       # Add LLM summaries via Claude Haiku → updates repos.parquet
python 02_embed_readmes.py            # Embed READMEs via Cohere → embeddings.npz
python 03_reduce_umap.py              # UMAP 512-dim → 2D → umap_coords.npz
python 04_label_topics.py             # Toponymy + Claude Sonnet topic labels → labels.parquet
python 05_visualize.py                # DataMapPlot interactive HTML → github_map.html
```

`04b_experiment.py` is a side branch for comparing embedder configurations (MiniLM vs Cohere), outputting to `data/experiments/`.

## Required Environment Variables

Set in `.env` (loaded by `python-dotenv`):
- `GITHUB_TOKEN` — used by `01_fetch_repos.py`
- `CO_API_KEY` — Cohere API key, used by `02_embed_readmes.py`, `04_label_topics.py`, `04b_experiment.py`
- `ANTHROPIC_API_KEY` — used by `01b_summarize_readmes.py`, `04_label_topics.py`, `04b_experiment.py`

## Architecture

**Sequential data pipeline** — each script reads outputs from previous stages:

```
repos.parquet ──┬──> embeddings.npz ──> umap_coords.npz ──> labels.parquet
                │                                                │
                └────────────────────────────────────────────────┴──> github_map.html
```

**`config.py`** is the central configuration hub: all file paths, API keys, and constants (batch sizes, model names, target counts) are defined here. Every pipeline script imports from it.

**Key technology choices:**
- Cohere `embed-v4.0` (512-dim) for README embeddings
- UMAP (n_neighbors=15, min_dist=0.05, cosine) for dimensionality reduction
- Toponymy library for hierarchical topic clustering with LLM-generated labels
- DataMapPlot for the final interactive HTML visualization
- Claude Haiku for README summarization, Claude Sonnet for topic naming

## Data Directory

All outputs go to `data/` (gitignored). Key files: `repos.parquet`, `embeddings.npz`, `umap_coords.npz`, `labels.parquet`, `toponymy_model.joblib`, `github_map.html`.

## Visualization Details

`05_visualize.py` produces the main output — an interactive HTML map with multiple colormaps (language, stars, license, age), hover tooltips with summaries, click-to-open-repo, and search. It uses coarse/fine topic labels for clustering display.
