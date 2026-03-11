"""Generate interactive DataMapPlot visualization."""

from datetime import datetime, timezone

import datamapplot
import glasbey
import numpy as np
import pandas as pd

from config import GITHUB_MAP_HTML, LABELS_PARQUET, REPOS_PARQUET, UMAP_COORDS_NPZ


def main():
    # Load data
    df = pd.read_parquet(REPOS_PARQUET)
    labels_df = pd.read_parquet(LABELS_PARQUET)
    coords = np.load(UMAP_COORDS_NPZ)["coords"]

    coarse_labels = labels_df["coarse_label"].values
    fine_labels = labels_df["fine_label"].values

    # ── Hover text ───────────────────────────────────────────────────────────
    hover_text = [
        f"{row['full_name']}\n⭐ {row['stargazers_count']:,} | {row['language'] or 'N/A'}"
        for _, row in df.iterrows()
    ]

    # ── Marker sizes (sqrt of stars) ─────────────────────────────────────────
    marker_sizes = np.sqrt(df["stargazers_count"].values).astype(float)
    # Normalize to reasonable pixel range
    marker_sizes = 3 + 15 * (marker_sizes - marker_sizes.min()) / (marker_sizes.max() - marker_sizes.min())

    # ── Colormap raw data ────────────────────────────────────────────────────

    # 1. Primary Language (categorical)
    languages = df["language"].fillna("").replace("", "Other").values
    unique_langs = sorted(set(languages))
    lang_palette = glasbey.create_palette(palette_size=len(unique_langs))
    lang_color_mapping = dict(zip(unique_langs, lang_palette))

    # 2. Star Count (continuous, log10)
    star_counts = np.log10(df["stargazers_count"].values.astype(float))

    # 3. License Type (categorical)
    licenses = df["license"].fillna("").replace("", "None").values
    unique_licenses = sorted(set(licenses))
    license_palette = glasbey.create_palette(palette_size=len(unique_licenses))
    license_color_mapping = dict(zip(unique_licenses, license_palette))

    # 4. Repo Age (continuous, years since creation)
    now = datetime.now(tz=timezone.utc)
    repo_ages = np.array(
        [(now - pd.to_datetime(d, utc=True).to_pydatetime()).days / 365.25 for d in df["created_at"]]
    )

    # ── Build the interactive plot ───────────────────────────────────────────
    fig = datamapplot.create_interactive_plot(
        coords,
        coarse_labels,
        fine_labels,
        hover_text=hover_text,
        marker_size_array=marker_sizes,
        colormap_rawdata=[languages, star_counts, licenses, repo_ages],
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
            {
                "field": "license",
                "description": "License Type",
                "kind": "categorical",
                "color_mapping": license_color_mapping,
            },
            {
                "field": "age",
                "description": "Repo Age (years)",
                "kind": "continuous",
                "cmap": "viridis",
            },
        ],
        darkmode=False,
    )
    fig.save(str(GITHUB_MAP_HTML))
    print(f"Saved interactive map to {GITHUB_MAP_HTML}")


if __name__ == "__main__":
    main()
