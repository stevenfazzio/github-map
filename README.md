# Semantic GitHub Map

**[View the live map](https://stevenfazzio.github.io/semantic-github-map/)**

An interactive 2D map of the top 10,000 most-starred GitHub repositories, positioned by semantic similarity of their READMEs.

The pipeline enumerates candidates via BigQuery, fetches repo metadata via GraphQL, embeds READMEs with Cohere, reduces to 2D with UMAP, generates topic labels with Claude, and renders an interactive HTML visualization with DataMapPlot.

## Pipeline

Each stage is a standalone script. Run them in order:

```bash
pip install -e .                      # Install dependencies
pip install -e '.[bigquery]'          # (optional) for 00_enumerate_repos.py

python 00_enumerate_repos.py          # BigQuery → data/candidates.csv (optional, see below)
python 01_fetch_repos.py              # GraphQL direct lookups → repos.parquet
python 01b_summarize_readmes.py       # Summarize READMEs via Claude Haiku → repos.parquet
python 02_embed_readmes.py            # Embed READMEs via Cohere → embeddings.npz
python 03_reduce_umap.py              # UMAP 512D → 2D → umap_coords.npz
python 04_label_topics.py             # Hierarchical topic labels via Toponymy + Claude → labels.parquet
python 05_visualize.py                # Interactive HTML map → github_map.html
```

**Two-phase fetch approach:** Step 00 queries GH Archive on BigQuery for repos with significant recent star activity, producing a generous candidate list (~25K). Step 01 then looks up each candidate via `repository(owner:, name:)` GraphQL queries (batched 50 per request), sorts by stars, and keeps the top 10K. This avoids the Search API's 1,000-result cap and non-deterministic ordering.

**Fallback for users without GCP:** A `candidates.csv` is committed to the repo root. If `data/candidates.csv` doesn't exist, `01_fetch_repos.py` copies from the committed file automatically. Most contributors can skip step 00 entirely.

Data flows through `data/` (gitignored):

```
candidates.csv ──> repos.parquet ──┬──> embeddings.npz ──> umap_coords.npz ──> labels.parquet
                                   │                                                │
                                   └────────────────────────────────────────────────┴──> github_map.html
```

## Environment Variables

Set in `.env`:

| Variable | Used by | Purpose |
|---|---|---|
| `GITHUB_TOKEN` | `01_fetch_repos.py` | GitHub API authentication |
| `CO_API_KEY` | `02_embed_readmes.py`, `04_label_topics.py` | Cohere embeddings |
| `ANTHROPIC_API_KEY` | `01b_summarize_readmes.py`, `04_label_topics.py` | Claude summarization & topic naming |
| `GCP_PROJECT` | `00_enumerate_repos.py` | (optional) Google Cloud project ID for BigQuery |

## Visualization Features

The output `github_map.html` includes:

- **Pan & zoom** across the semantic map
- **Hover tooltips** with repo name, stars, language, and summary
- **Click to open** any repo on GitHub
- **Search** repos by name
- **Nine colormaps**: primary language, star count, license type, license family, created date, owner type, last push, fork count, open issues
- **Hierarchical topic labels** at multiple levels of detail for cluster display
- **Methodology page** (`methodology.html`) linked from the nav bar

## Technical Details

| Component | Choice |
|---|---|
| Repo enumeration | BigQuery (GH Archive) |
| Repo metadata | GitHub GraphQL API (`repository()` direct lookups, batched 50/request) |
| Embeddings | Cohere `embed-v4.0` (512-dim) |
| Dimensionality reduction | UMAP (n_neighbors=15, min_dist=0.05, cosine) 512D → 2D |
| Topic clustering | [Toponymy](https://github.com/TutteInstitute/Toponymy) (hierarchical density-based) |
| Topic naming | Claude Sonnet |
| README summarization | Claude Haiku |
| Visualization | [DataMapPlot](https://github.com/TutteInstitute/DataMapPlot) |

## Requirements

Python ≥ 3.10. Key dependencies: `cohere`, `umap-learn`, `toponymy`, `datamapplot`, `anthropic`, `pandas`, `numpy`.

See `pyproject.toml` for the full list.
