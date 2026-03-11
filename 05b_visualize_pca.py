"""Generate DataMapPlot maps for each PCA experiment variation."""

from datetime import datetime, timezone
from pathlib import Path

import datamapplot
import glasbey
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

from config import EMBEDDINGS_NPZ, PCA_EXPERIMENTS_DIR, REPOS_PARQUET


EXPERIMENTS = [
    {"name": "baseline_no_pca", "pca_dims": None},
    {"name": "pca_250", "pca_dims": 250},
    {"name": "pca_125", "pca_dims": 125},
    {"name": "pca_50", "pca_dims": 50},
]


def ensure_coords(exp_dir, embeddings, pca_dims):
    """Load or compute UMAP coords for an experiment."""
    coords_path = exp_dir / "umap_coords.npz"
    if coords_path.exists():
        return np.load(coords_path)["coords"]

    print(f"  Computing coords (PCA dims: {pca_dims or 'None'})...")
    if pca_dims is not None:
        pca = PCA(n_components=pca_dims, random_state=42)
        umap_input = pca.fit_transform(embeddings)
    else:
        umap_input = embeddings

    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.05,
        metric="cosine", random_state=42,
    )
    coords = reducer.fit_transform(umap_input)
    np.savez(coords_path, coords=coords)
    return coords


def make_map(df, coords, coarse_labels, fine_labels, output_path, title_suffix=""):
    """Build and save a DataMapPlot interactive HTML map."""
    has_summary = "summary" in df.columns
    hover_text = [
        f"{row['full_name']}\n⭐ {row['stargazers_count']:,} | {row['language'] or 'N/A'}"
        + (f"\n\n{row['summary']}" if has_summary and row.get("summary") else "")
        for _, row in df.iterrows()
    ]

    marker_sizes = np.sqrt(df["stargazers_count"].values).astype(float)
    marker_sizes = 3 + 15 * (marker_sizes - marker_sizes.min()) / (marker_sizes.max() - marker_sizes.min())

    # Primary Language
    raw_languages = df["language"].fillna("Other").replace("", "Other")
    non_other = raw_languages[raw_languages != "Other"]
    top_languages = non_other.value_counts().head(9).index.tolist()
    languages = raw_languages.where(raw_languages.isin(top_languages), "Other").values
    unique_langs = sorted(set(languages))
    lang_palette = glasbey.create_palette(palette_size=len(unique_langs))
    lang_color_mapping = dict(zip(unique_langs, lang_palette))

    # Star Count
    star_counts = np.log10(df["stargazers_count"].values.astype(float))

    extra_data = pd.DataFrame({"full_name": df["full_name"].values})

    fig = datamapplot.create_interactive_plot(
        coords,
        coarse_labels,
        fine_labels,
        hover_text=hover_text,
        marker_size_array=marker_sizes,
        extra_point_data=extra_data,
        on_click="window.open(`https://github.com/{full_name}`, '_blank')",
        colormap_rawdata=[languages, star_counts],
        colormap_metadata=[
            {
                "field": "language",
                "description": "Primary Language",
                "kind": "categorical",
                "color_mapping": lang_color_mapping,
            },
            {
                "field": "stars",
                "description": "Star Count (log10)",
                "kind": "continuous",
                "cmap": "YlOrRd",
            },
        ],
        enable_search=True,
        darkmode=False,
    )
    fig.save(str(output_path))
    print(f"  Saved map to {output_path}")


def main():
    df = pd.read_parquet(REPOS_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    print(f"Loaded {len(df)} repos")

    for exp in EXPERIMENTS:
        name = exp["name"]
        exp_dir = PCA_EXPERIMENTS_DIR / name
        labels_path = exp_dir / "labels.parquet"

        if not labels_path.exists():
            print(f"Skipping {name}: no labels.parquet found")
            continue

        print(f"\nGenerating map: {name}")
        labels_df = pd.read_parquet(labels_path)
        coords = ensure_coords(exp_dir, embeddings, exp["pca_dims"])

        make_map(
            df, coords,
            labels_df["coarse_label"].values,
            labels_df["fine_label"].values,
            exp_dir / "github_map.html",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
