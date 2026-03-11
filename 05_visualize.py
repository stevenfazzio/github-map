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

    # 4. License Family (categorical, grouped from License Type)
    license_to_family = {
        "AGPL-3.0": "GPL",
        "GPL-2.0": "GPL",
        "GPL-3.0": "GPL",
        "LGPL-3.0": "GPL",
        "BSD-2-Clause": "BSD",
        "BSD-3-Clause": "BSD",
        "CC-BY-4.0": "Creative Commons",
        "CC-BY-SA-4.0": "Creative Commons",
        "CC0-1.0": "Creative Commons",
        "Apache-2.0": "Apache",
        "MIT": "MIT",
        "MPL-2.0": "MPL",
        "ISC": "Other Permissive",
        "Unlicense": "Other Permissive",
        "WTFPL": "Other Permissive",
        "Zlib": "Other Permissive",
        "Vim": "Other Permissive",
        "OFL-1.1": "Other Permissive",
        "NOASSERTION": "Unknown/None",
        "None": "Unknown/None",
    }
    license_families = np.array([license_to_family.get(l, "Unknown/None") for l in licenses])
    unique_families = sorted(set(license_families))
    family_palette = glasbey.create_palette(palette_size=len(unique_families))
    family_color_mapping = dict(zip(unique_families, family_palette))

    # 5. Repo Age (continuous, years since creation)
    now = datetime.now(tz=timezone.utc)
    repo_ages = np.array(
        [(now - pd.to_datetime(d, utc=True).to_pydatetime()).days / 365.25 for d in df["created_at"]]
    )

    # ── Build the interactive plot ───────────────────────────────────────────
    extra_data = pd.DataFrame({"full_name": df["full_name"].values})

    fig = datamapplot.create_interactive_plot(
        coords,
        coarse_labels,
        fine_labels,
        hover_text=hover_text,
        marker_size_array=marker_sizes,
        extra_point_data=extra_data,
        on_click="window.open(`https://github.com/{full_name}`, '_blank')",
        colormap_rawdata=[languages, star_counts, licenses, license_families, repo_ages],
        colormap_metadata=[
            {
                "field": "language",
                "description": "Primary Language",
                "kind": "categorical",
                "color_mapping": lang_color_mapping,
                "show_legend": True,
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
                "field": "license_family",
                "description": "License Family",
                "kind": "categorical",
                "color_mapping": family_color_mapping,
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
