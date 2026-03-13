"""Generate simplified DataMapPlot visualizations for each experiment.

Uses the coarse_label and fine_label from each experiment's labels.parquet
to produce a quick comparison map (no extra colormaps or nav bar).
"""

import numpy as np
import pandas as pd
import datamapplot

from config import EXPERIMENTS_DIR, REPOS_PARQUET, UMAP_COORDS_NPZ


EXPERIMENTS = [
    "detail_full_range",
    "detail_medium",
    "detail_concise",
    "detail_broad",
]


def main():
    df = pd.read_parquet(REPOS_PARQUET)
    coords = np.load(UMAP_COORDS_NPZ)["coords"]

    # Shared hover text (lightweight)
    hover_text = []
    for _, row in df.iterrows():
        stars = f"{row['stargazers_count']:,}"
        lang = row["language"] or "N/A"
        hover_text.append(f"{row['full_name']}\n{stars} stars | {lang}")

    marker_sizes = np.sqrt(df["stargazers_count"].values.astype(float))
    marker_sizes = 3 + 15 * (marker_sizes - marker_sizes.min()) / (marker_sizes.max() - marker_sizes.min())

    for name in EXPERIMENTS:
        exp_dir = EXPERIMENTS_DIR / name
        labels_path = exp_dir / "labels.parquet"
        if not labels_path.exists():
            print(f"Skipping {name}: no labels.parquet")
            continue

        labels_df = pd.read_parquet(labels_path)
        coarse = labels_df["coarse_label"].values
        fine = labels_df["fine_label"].values

        print(f"Generating map for {name}...")
        fig = datamapplot.create_interactive_plot(
            coords,
            coarse,
            fine,
            hover_text=hover_text,
            marker_size_array=marker_sizes,
            title=f"GitHub Map — {name}",
            sub_title=f"Detail level experiment: {name}",
            enable_search=True,
            darkmode=False,
        )
        out_path = exp_dir / "map.html"
        fig.save(str(out_path))
        print(f"  Saved to {out_path}")

    print("\nDone. Open any map.html in a browser to compare.")


if __name__ == "__main__":
    main()
