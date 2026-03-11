# GitHub Map

An interactive 2D map of the top 1,000 most-starred GitHub repositories, positioned by semantic similarity of their READMEs.

The pipeline fetches repo metadata, embeds READMEs with Cohere, reduces to 2D with PCA + UMAP, generates topic labels with Claude, and renders an interactive HTML visualization with DataMapPlot.

## Pipeline

Each stage is a standalone script. Run them in order:

```bash
pip install -e .

python 01_fetch_repos.py          # Fetch top 1K repos → repos.parquet
python 01b_summarize_readmes.py   # Summarize READMEs via Claude Haiku → repos.parquet
python 02_embed_readmes.py        # Embed READMEs via Cohere → embeddings.npz
python 03_reduce_umap.py          # PCA 512→256, UMAP 256→2D → umap_coords.npz
python 04_label_topics.py         # Hierarchical topic labels via Toponymy + Claude → labels.parquet
python 05_visualize.py            # Interactive HTML map → github_map.html
```

Data flows through `data/` (gitignored):

```
repos.parquet → embeddings.npz → umap_coords.npz → labels.parquet
       │                                                  │
       └──────────────────────────────────────────────────┴→ github_map.html
```

## Environment Variables

Set in `.env`:

| Variable | Used by | Purpose |
|---|---|---|
| `GITHUB_TOKEN` | `01_fetch_repos.py` | GitHub API authentication |
| `CO_API_KEY` | `02_embed_readmes.py`, `04_label_topics.py` | Cohere embeddings |
| `ANTHROPIC_API_KEY` | `01b_summarize_readmes.py`, `04_label_topics.py` | Claude summarization & topic naming |

## Visualization Features

The output `github_map.html` includes:

- **Pan & zoom** across the semantic map
- **Hover tooltips** with repo name, stars, language, and summary
- **Click to open** any repo on GitHub
- **Search** repos by name
- **Five colormaps**: primary language, star count, license type, license family, repo age
- **Two-level topic labels** (coarse and fine) for cluster display
- **Methodology page** (`methodology.html`) linked from the nav bar

## Technical Details

| Component | Choice |
|---|---|
| Embeddings | Cohere `embed-v4.0` (512-dim) |
| Dimensionality reduction | PCA 512→256, then UMAP (n_neighbors=15, min_dist=0.05, cosine) |
| Topic clustering | [Toponymy](https://github.com/TutteInstitute/Toponymy) (hierarchical density-based) |
| Topic naming | Claude Sonnet |
| README summarization | Claude Haiku |
| Visualization | [DataMapPlot](https://github.com/TutteInstitute/DataMapPlot) |

## Requirements

Python ≥ 3.10. Key dependencies: `cohere`, `umap-learn`, `toponymy`, `datamapplot`, `anthropic`, `pandas`, `numpy`.

See `pyproject.toml` for the full list.
