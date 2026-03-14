"""Generate interactive DataMapPlot visualization."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import datamapplot
import glasbey
import numpy as np
import pandas as pd

from config import (
    DOCS_INDEX_HTML,
    DOCS_METHODOLOGY_HTML,
    GITHUB_MAP_HTML,
    LABELS_PARQUET,
    METHODOLOGY_HTML,
    REPOS_PARQUET,
    UMAP_COORDS_NPZ,
)

FILTER_PANEL_HTML = Path(__file__).parent / "filter_panel.html"

# Custom JS injected at render time: hide boundaries immediately after async
# creation, and patch pointLayer.clone to preserve radiusMinPixels/radiusMaxPixels.
CUSTOM_JS = """
// Monkey-patch addBoundaries so boundaries start hidden
(function() {
  var origAddBoundaries = datamap.addBoundaries.bind(datamap);
  datamap.addBoundaries = function() {
    origAddBoundaries.apply(this, arguments);
    // Hide the boundary layer immediately
    if (datamap.boundaryLayer) {
      var idx = datamap.layers.indexOf(datamap.boundaryLayer);
      if (idx !== -1) {
        var replacement = new deck.PolygonLayer({
          id: 'boundaryLayer',
          data: datamap.boundaryLayer.props.data,
          stroked: true,
          filled: false,
          getLineColor: function(d) { return [d.r, d.g, d.b, d.a]; },
          getPolygon: function(d) { return d.polygon; },
          lineWidthUnits: 'common',
          getLineWidth: function(d) { return d.size * d.size; },
          lineWidthScale: datamap.clusterBoundaryLineWidth * 5e-5,
          lineJointRounded: true,
          lineWidthMaxPixels: 4,
          lineWidthMinPixels: 0.0,
          parameters: { depthTest: false },
          visible: false,
        });
        datamap.layers[idx] = replacement;
        datamap.boundaryLayer = replacement;
        datamap.deckgl.setProps({ layers: [].concat(datamap.layers) });
      }
    }
  };
})();
"""


def main():
    # Load data
    df = pd.read_parquet(REPOS_PARQUET)
    labels_df = pd.read_parquet(LABELS_PARQUET)
    coords = np.load(UMAP_COORDS_NPZ)["coords"]

    # Collect all label layers (label_layer_0 = coarsest, label_layer_1, ... = finer)
    label_columns = sorted(c for c in labels_df.columns if c.startswith("label_layer_"))
    topic_name_vectors = [labels_df[c].values for c in label_columns]

    # ── Hover text ───────────────────────────────────────────────────────────
    has_summary = "summary" in df.columns
    has_forks = "fork_count" in df.columns
    hover_text = []
    for _, row in df.iterrows():
        line1 = row["full_name"]
        stars = f"⭐ {row['stargazers_count']:,}"
        forks = f" | 🍴 {row['fork_count']:,}" if has_forks else ""
        lang = row["language"] or "N/A"
        line2 = f"{stars}{forks} | {lang}"
        text = f"{line1}\n{line2}"
        if has_summary and row.get("summary"):
            text += f"\n\n{row['summary']}"
        hover_text.append(text)

    # ── Marker sizes (sqrt of stars) ─────────────────────────────────────────
    marker_sizes = np.sqrt(df["stargazers_count"].values).astype(float)
    # Normalize to reasonable pixel range
    marker_sizes = 3 + 15 * (marker_sizes - marker_sizes.min()) / (marker_sizes.max() - marker_sizes.min())

    # ── Colormap raw data ────────────────────────────────────────────────────

    # 1. Primary Language (categorical, top 9 + Other)
    raw_languages = df["language"].fillna("Other").replace("", "Other")
    non_other = raw_languages[raw_languages != "Other"]
    top_languages = non_other.value_counts().head(9).index.tolist()
    languages = raw_languages.where(raw_languages.isin(top_languages), "Other").values
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

    # 5. Repo Created Date (datetime, for continuous colormap)
    now = datetime.now(tz=timezone.utc)
    created_dates = pd.to_datetime(df["created_at"], utc=True).values

    # 6–9. New colormaps from GraphQL fields (guarded for backward compat)
    extra_rawdata = []
    extra_metadata = []

    if "owner_type" in df.columns:
        owner_types = df["owner_type"].fillna("Unknown").replace("", "Unknown").values
        unique_owners = sorted(set(owner_types))
        owner_palette = glasbey.create_palette(palette_size=len(unique_owners))
        owner_color_mapping = dict(zip(unique_owners, owner_palette))
        extra_rawdata.append(owner_types)
        extra_metadata.append({
            "field": "owner_type",
            "description": "Owner Type",
            "kind": "categorical",
            "color_mapping": owner_color_mapping,
        })

    if "pushed_at" in df.columns:
        days_since_push = np.array([
            (now - pd.to_datetime(d, utc=True).to_pydatetime()).days
            if d else 9999
            for d in df["pushed_at"]
        ], dtype=float)
    else:
        days_since_push = np.zeros(len(df), dtype=float)

    if "pushed_at" in df.columns:
        extra_rawdata.append(days_since_push)
        extra_metadata.append({
            "field": "last_push",
            "description": "Last Push (days ago)",
            "kind": "continuous",
            "cmap": "RdYlGn_r",
        })

    if "fork_count" in df.columns:
        fork_counts_log = np.log10(df["fork_count"].values.astype(float).clip(min=1))
        extra_rawdata.append(fork_counts_log)
        extra_metadata.append({
            "field": "forks",
            "description": "Fork Count (log10)",
            "kind": "continuous",
            "cmap": "YlGnBu",
        })

    if "open_issue_count" in df.columns:
        open_issues_log = np.log10(df["open_issue_count"].values.astype(float).clip(min=1))
        extra_rawdata.append(open_issues_log)
        extra_metadata.append({
            "field": "open_issues",
            "description": "Open Issues (log10)",
            "kind": "continuous",
            "cmap": "OrRd",
        })

    # ── Build the interactive plot ───────────────────────────────────────────

    # Extra point data for filters (all as strings for DataMapPlot serialization)
    created_years = pd.to_datetime(df["created_at"], utc=True).dt.year.values
    summaries = df["summary"].fillna("").values if "summary" in df.columns else [""] * len(df)
    search_text = [
        f"{fn} {lang or ''} {summ}"
        for fn, lang, summ in zip(df["full_name"], df["language"].fillna(""), summaries)
    ]

    extra_data = pd.DataFrame({
        "full_name": df["full_name"].values,
        "stars": df["stargazers_count"].astype(str).values,
        "created_year": created_years.astype(str),
        "days_since_push": days_since_push.astype(int).astype(str) if "pushed_at" in df.columns else "0",
        "forks": df["fork_count"].astype(str).values if "fork_count" in df.columns else "0",
        "open_issues": df["open_issue_count"].astype(str).values if "open_issue_count" in df.columns else "0",
        "primary_language": languages,
        "search_text": search_text,
    })

    all_rawdata = [languages, star_counts, licenses, license_families, created_dates] + extra_rawdata
    all_metadata = [
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
                "field": "license_family",
                "description": "License Family",
                "kind": "categorical",
                "color_mapping": family_color_mapping,
            },
            {
                "field": "created",
                "description": "Created Date",
                "kind": "datetime",
                "cmap": "viridis",
            },
    ] + extra_metadata

    fig = datamapplot.create_interactive_plot(
        coords,
        *topic_name_vectors,
        hover_text=hover_text,
        marker_size_array=marker_sizes,
        extra_point_data=extra_data,
        on_click="window.open(`https://github.com/{full_name}`, '_blank')",
        colormap_rawdata=all_rawdata,
        colormap_metadata=all_metadata,
        title="GitHub Map",
        sub_title="Top 10,000 most-starred repositories, mapped by README similarity",
        enable_search=True,
        search_field="search_text",
        custom_js=CUSTOM_JS,
        cluster_boundary_polygons=True,
        cluster_boundary_line_width=1.0,
        darkmode=False,
    )
    fig.save(str(GITHUB_MAP_HTML))
    print(f"Saved interactive map to {GITHUB_MAP_HTML}")

    _inject_nav(GITHUB_MAP_HTML)
    print("Injected navigation bar into map")

    _inject_filters(GITHUB_MAP_HTML, df, languages)
    print("Injected filter panel into map")

    _write_methodology(METHODOLOGY_HTML)
    print(f"Saved methodology page to {METHODOLOGY_HTML}")

    # ── Write docs/ copies for GitHub Pages ───────────────────────────────────
    _copy_for_docs(GITHUB_MAP_HTML, DOCS_INDEX_HTML)
    _write_methodology(DOCS_METHODOLOGY_HTML, map_href="index.html")
    print(f"Saved docs/ copies for GitHub Pages")


# ── Filter panel injection ───────────────────────────────────────────────────


def _inject_filters(html_path, df, languages):
    """Inject the advanced filter panel into DataMapPlot-generated HTML."""
    now = datetime.now(tz=timezone.utc)

    html = Path(html_path).read_text()

    # 1. Dispatch datamapReady event after metadata finishes loading
    html = html.replace(
        "updateProgressBar('meta-data-progress', 100);\n      checkAllDataLoaded();",
        "updateProgressBar('meta-data-progress', 100);\n"
        "      window.dispatchEvent(new CustomEvent('datamapReady', "
        "{ detail: { datamap, hoverData } }));\n"
        "      checkAllDataLoaded();",
        1,
    )

    # 2. CSS fixes for content-wrapper (runs after _inject_nav, no conflicts)
    html = html.replace(
        "height:100%;z-index:1;padding:0",
        "height:100vh;z-index:1;box-sizing:border-box;padding:0",
        1,
    )
    html = html.replace(
        "grid-template-rows:1fr 1fr",
        "grid-template-rows:minmax(0,1fr) minmax(0,1fr)",
        1,
    )

    # 3. Compute filter config
    stars = df["stargazers_count"].values.astype(int)
    created_years = pd.to_datetime(df["created_at"], utc=True).dt.year.values.astype(int)

    if "pushed_at" in df.columns:
        days = np.array([
            (now - pd.to_datetime(d, utc=True).to_pydatetime()).days
            if d else 9999
            for d in df["pushed_at"]
        ], dtype=int)
    else:
        days = np.zeros(len(df), dtype=int)

    forks = df["fork_count"].values.astype(int) if "fork_count" in df.columns else np.zeros(len(df), dtype=int)
    issues = df["open_issue_count"].values.astype(int) if "open_issue_count" in df.columns else np.zeros(len(df), dtype=int)

    def p99_cap(arr):
        return int(np.percentile(arr, 99))

    # Build sorted language list: top 9 + Other (matching the colormap logic)
    unique_langs = sorted(set(languages))
    sorted_language_list = [l for l in unique_langs if l != "Other"] + ["Other"]

    filter_config = {
        "totalCount": len(df),
        "ranges": {
            "stars": {"min": int(stars.min()), "max": int(stars.max()), "sliderMax": p99_cap(stars)},
            "created_year": {"min": int(created_years.min()), "max": int(created_years.max()), "sliderMax": int(created_years.max())},
            "days_since_push": {"min": 0, "max": int(days.max()), "sliderMax": p99_cap(days)},
            "forks": {"min": 0, "max": int(forks.max()), "sliderMax": p99_cap(forks)},
            "open_issues": {"min": 0, "max": int(issues.max()), "sliderMax": p99_cap(issues)},
        },
        "languages": sorted_language_list,
    }

    # 4. Read and split template by <!-- SECTION: xxx --> markers
    template = FILTER_PANEL_HTML.read_text()
    sections = re.split(r"<!-- SECTION: (\w+) -->", template)
    # sections = ['', 'css', '<css content>', 'html', '<html content>', 'js', '<js content>']
    section_map = {}
    for i in range(1, len(sections), 2):
        section_map[sections[i]] = sections[i + 1].strip()

    # 5. Replace config placeholder in JS section
    js_section = section_map["js"].replace("__FILTER_CONFIG_JSON__", json.dumps(filter_config))

    # 6. Inject CSS before </head>
    html = html.replace("</head>", section_map["css"] + "\n</head>", 1)

    # 7. Inject HTML after search-container div
    search_pattern = re.compile(
        r'(<div id="search-container" class="container-box[^"]*">\s*'
        r'<input[^/]*/>\s*</div>)'
    )
    match = search_pattern.search(html)
    if match:
        insert_pos = match.end()
        html = html[:insert_pos] + "\n      " + section_map["html"] + "\n" + html[insert_pos:]

    # 8. Inject JS before </html>
    html = html.replace("</html>", js_section + "\n</html>", 1)

    Path(html_path).write_text(html)


# ── Nav bar injection ────────────────────────────────────────────────────────


def _copy_for_docs(src_html_path, dest_html_path):
    """Copy HTML file to docs/, replacing github_map.html links with index.html."""
    html = Path(src_html_path).read_text()
    html = html.replace('href="github_map.html"', 'href="index.html"')
    Path(dest_html_path).write_text(html)


def _inject_nav(html_path):
    """Add site navigation bar to DataMapPlot-generated HTML."""
    html = Path(html_path).read_text()

    nav_css = """<style>
.site-nav{position:fixed;top:0;left:0;right:0;z-index:200;
  background:rgba(255,255,255,0.85);backdrop-filter:blur(8px);
  -webkit-backdrop-filter:blur(8px);border-bottom:1px solid #e0e0e0;
  padding:0 24px;height:44px;display:flex;align-items:center;gap:24px;
  font-family:system-ui,sans-serif;font-size:14px;font-weight:500;pointer-events:auto;}
.site-nav a{color:#333;text-decoration:none;transition:color 0.15s;}
.site-nav a:hover{color:#1a73e8;}
.site-nav a.active{color:#1a73e8;border-bottom:2px solid #1a73e8;line-height:42px;}
</style>"""

    nav_html = """<nav class="site-nav">
  <a href="github_map.html" class="active">Visualization</a>
  <a href="methodology.html">Methodology</a>
</nav>"""

    # Inject after <body> tag
    html = html.replace("<body>", f"<body>{nav_css}{nav_html}", 1)

    # Offset the fixed deck-container below the nav bar
    html = html.replace(
        "position: fixed; z-index: -1; top: 0; left: 0; width: 100%; height: 100%;",
        "position: fixed; z-index: -1; top: 44px; left: 0; width: 100%; height: calc(100% - 44px);",
        1,
    )

    # Shrink body and content to account for nav bar height
    html = html.replace(
        "overflow: hidden;",
        "overflow: hidden; padding-top: 44px; height: calc(100vh - 44px);",
        1,
    )
    html = html.replace(
        "height: 100vh;",
        "height: calc(100vh - 44px);",
        1,
    )
    # Adjust content-wrapper min-height so bottom-left stays in viewport
    html = html.replace(
        "min-height:calc(100vh - 16px)",
        "min-height:calc(100vh - 60px)",
        1,
    )

    Path(html_path).write_text(html)


# ── Methodology page ─────────────────────────────────────────────────────────


def _write_methodology(output_path, map_href="github_map.html"):
    """Write standalone methodology HTML page."""
    html = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Methodology — GitHub Map</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Roboto', system-ui, sans-serif;
  color: #000;
  background: #fff;
  line-height: 1.7;
  font-size: 16px;
}
.site-nav {
  position: fixed; top: 0; left: 0; right: 0; z-index: 200;
  background: rgba(255,255,255,0.85); backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px); border-bottom: 1px solid #e0e0e0;
  padding: 0 24px; height: 44px; display: flex; align-items: center; gap: 24px;
  font-size: 14px; font-weight: 500; pointer-events: auto;
}
.site-nav a { color: #333; text-decoration: none; transition: color 0.15s; }
.site-nav a:hover { color: #1a73e8; }
.site-nav a.active { color: #1a73e8; border-bottom: 2px solid #1a73e8; line-height: 42px; }
.content { max-width: 720px; margin: 0 auto; padding: 68px 24px 80px; }
h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
.subtitle { color: #555; font-size: 16px; margin-bottom: 40px; }
h2 {
  font-size: 20px; font-weight: 700; margin-top: 48px; margin-bottom: 16px;
  padding-bottom: 6px; border-bottom: 1px solid #e0e0e0; scroll-margin-top: 72px;
}
h3 { font-size: 16px; font-weight: 700; margin-top: 24px; margin-bottom: 8px; }
p { margin-bottom: 16px; }
a { color: #1a73e8; text-decoration: none; }
a:hover { text-decoration: underline; }
ul, ol { margin-bottom: 16px; padding-left: 24px; }
li { margin-bottom: 6px; }
table { width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 15px; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #e0e0e0; }
th { font-weight: 500; color: #555; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
code { font-size: 14px; background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }
.pipeline-step { margin-bottom: 12px; padding-left: 16px; border-left: 3px solid #e0e0e0; }
.pipeline-step strong { display: block; margin-bottom: 2px; }
.pipeline-step p { margin-bottom: 0; color: #333; font-size: 15px; }
.note { background: #f8f9fa; border-left: 3px solid #1a73e8; padding: 12px 16px; margin: 16px 0; font-size: 14px; color: #333; }
.toc { margin-bottom: 32px; padding-bottom: 16px; border-bottom: 1px solid #e0e0e0; }
.toc ul { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 8px 16px; }
.toc li { margin: 0; }
.toc a { font-size: 14px; font-weight: 500; color: #555; text-decoration: none; transition: color 0.15s; }
.toc a:hover { color: #1a73e8; text-decoration: none; }
.toc a.active { color: #1a73e8; }
@media (min-width: 1100px) {
  .content { margin-left: calc(50% - 260px); margin-right: auto; }
  .toc { position: fixed; top: 68px; left: calc(50% - 444px); width: 160px; border-bottom: none; padding-bottom: 0; margin-bottom: 0; }
  .toc ul { display: block; }
  .toc li { margin-bottom: 12px; }
  .toc a { display: block; padding-left: 12px; border-left: 2px solid transparent; line-height: 1.4; }
  .toc a.active { color: #1a73e8; border-left-color: #1a73e8; }
}
</style>
</head>
<body>
<nav class="site-nav">
  <a href="github_map.html">Visualization</a>
  <a href="methodology.html" class="active">Methodology</a>
</nav>
<div class="content">

<h1>Methodology</h1>
<p class="subtitle">How the GitHub Map visualization is built</p>

<nav class="toc">
  <ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#corpus-collection">Corpus Collection</a></li>
    <li><a href="#processing-pipeline">Processing Pipeline</a></li>
    <li><a href="#tools">Tools &amp; Technologies</a></li>
    <li><a href="#using-the-visualization">Using the Visualization</a></li>
    <li><a href="#field-definitions">Field Definitions</a></li>
  </ul>
</nav>

<h2 id="overview">Overview</h2>
<p>
  This visualization maps the top 10,000 most-starred GitHub repositories onto a 2D plane,
  positioned by the semantic similarity of their README files. Repositories with similar
  descriptions and purposes appear near each other, revealing natural clusters of related
  projects across the open-source ecosystem.
</p>
<p>
  The map is generated by an automated pipeline that fetches repository data from the GitHub
  API, embeds README content into high-dimensional vectors, reduces those vectors to two
  dimensions, applies hierarchical topic clustering, and renders an interactive HTML
  visualization.
</p>

<h2 id="corpus-collection">Corpus Collection</h2>
<p>
  Repositories are sourced from the GitHub Search API, ranked by star count in descending
  order. The pipeline fetches the top 10,000 repositories, collecting metadata (name, stars,
  language, license, creation date) and the full README content for each.
</p>
<p>
  Repositories whose READMEs are shorter than 200 characters after stripping whitespace are
  excluded before embedding. These include empty READMEs, placeholder files, and monorepo
  pointers that lack sufficient content for meaningful embedding and map placement.
</p>
<p>
  Each README is then summarized into a concise description using Claude Haiku, providing
  clean hover-text for the visualization and a normalized input for downstream embedding.
</p>

<h2 id="processing-pipeline">Processing Pipeline</h2>

<div class="pipeline-step">
  <strong>1. Fetch repositories</strong>
  <p><code>01_fetch_repos.py</code> &mdash; Queries the GitHub GraphQL API for the top 10,000
  most-starred repos, fetching metadata and README content in a single query per page,
  and saves everything to <code>repos.parquet</code>.</p>
</div>

<div class="pipeline-step">
  <strong>2. Summarize READMEs</strong>
  <p><code>01b_summarize_readmes.py</code> &mdash; Sends each README to Claude Haiku to produce
  a short natural-language summary, stored back into <code>repos.parquet</code>.</p>
</div>

<div class="pipeline-step">
  <strong>3. Embed READMEs</strong>
  <p><code>02_embed_readmes.py</code> &mdash; Encodes README text into 512-dimensional vectors
  using Cohere's <code>embed-v4.0</code> model, saved to <code>embeddings.npz</code>.</p>
</div>

<div class="pipeline-step">
  <strong>4. Reduce to 2D</strong>
  <p><code>03_reduce_umap.py</code> &mdash; Applies UMAP (n_neighbors=15, min_dist=0.05, cosine metric)
  to project from 512D to 2D coordinates in <code>umap_coords.npz</code>.</p>
</div>

<div class="pipeline-step">
  <strong>5. Label topics</strong>
  <p><code>04_label_topics.py</code> &mdash; Uses the Toponymy library for hierarchical density-based
  clustering, then sends representative documents from each cluster to Claude Sonnet
  to generate human-readable topic labels at multiple levels of detail, saved to <code>labels.parquet</code>.</p>
</div>

<div class="pipeline-step">
  <strong>6. Visualize</strong>
  <p><code>05_visualize.py</code> &mdash; Combines coordinates, labels, and metadata into an
  interactive HTML map using DataMapPlot, with multiple colormaps, search, hover tooltips,
  and click-to-open functionality.</p>
</div>

<h2 id="tools">Tools &amp; Technologies</h2>
<table>
  <thead>
    <tr><th>Tool</th><th>Role</th></tr>
  </thead>
  <tbody>
    <tr><td><a href="https://docs.cohere.com/reference/embed">Cohere <code>embed-v4.0</code></a></td><td>README text embedding (512 dimensions)</td></tr>
    <tr><td><a href="https://github.com/lmcinnes/umap">UMAP</a></td><td>Dimensionality reduction from 512D to 2D</td></tr>
    <tr><td><a href="https://github.com/TutteInstitute/toponymy">Toponymy</a></td><td>Hierarchical density-based topic clustering</td></tr>
    <tr><td><a href="https://github.com/TutteInstitute/datamapplot">DataMapPlot</a></td><td>Interactive HTML map rendering</td></tr>
    <tr><td><a href="https://www.anthropic.com/claude/haiku">Claude Haiku</a></td><td>README summarization</td></tr>
    <tr><td><a href="https://www.anthropic.com/claude/sonnet">Claude Sonnet</a></td><td>Topic label generation</td></tr>
    <tr><td><a href="https://docs.github.com/en/graphql">GitHub GraphQL API</a></td><td>Repository metadata and README fetching</td></tr>
  </tbody>
</table>

<h2 id="using-the-visualization">Using the Visualization</h2>
<ul>
  <li><strong>Pan and zoom</strong> &mdash; Click and drag to pan; scroll to zoom in and out.</li>
  <li><strong>Hover</strong> &mdash; Hover over any point to see the repository name, star count,
  primary language, and a short summary.</li>
  <li><strong>Click</strong> &mdash; Click any point to open the repository on GitHub in a new tab.</li>
  <li><strong>Search</strong> &mdash; Use the search box to find specific repositories by name.</li>
  <li><strong>Colormaps</strong> &mdash; Switch between colormaps using the dropdown to color points
  by language, star count, license, creation date, owner type, activity recency, fork count, or open issues.</li>
  <li><strong>Topic labels</strong> &mdash; Cluster labels appear on the map at multiple levels of detail,
  from broad categories down to specific sub-topics. The number of levels is chosen
  automatically by the clustering algorithm.</li>
</ul>

<h2 id="field-definitions">Field Definitions</h2>
<table>
  <thead>
    <tr><th>Colormap</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td>Primary Language</td><td>The repository's primary programming language as reported by GitHub. The top 9 languages are shown individually; all others are grouped as "Other".</td></tr>
    <tr><td>Star Count (log10)</td><td>Base-10 logarithm of the repository's star count. Log scale helps distinguish differences among highly-starred repos.</td></tr>
    <tr><td>License Type</td><td>The specific license identified by GitHub (e.g., MIT, Apache-2.0, GPL-3.0). Repos without a detected license show as "None".</td></tr>
    <tr><td>License Family</td><td>Licenses grouped into families: MIT, Apache, GPL, BSD, Creative Commons, MPL, Other Permissive, and Unknown/None.</td></tr>
    <tr><td>Created Date</td><td>The date the repository was created on GitHub. Older repos appear blue; newer repos appear yellow.</td></tr>
    <tr><td>Owner Type</td><td>Whether the repository is owned by an individual User or an Organization account.</td></tr>
    <tr><td>Last Push (days ago)</td><td>Days since the most recent push to the repository. Green indicates recent activity; red indicates staleness.</td></tr>
    <tr><td>Fork Count (log10)</td><td>Base-10 logarithm of the repository's fork count. Log scale helps distinguish differences among heavily-forked repos.</td></tr>
    <tr><td>Open Issues (log10)</td><td>Base-10 logarithm of the repository's open issue count. Log scale normalizes the wide range of issue counts.</td></tr>
  </tbody>
</table>

</div>

<script>
(function() {
  var tocLinks = document.querySelectorAll('.toc a');
  var sections = [];
  tocLinks.forEach(function(link) {
    var id = link.getAttribute('href').slice(1);
    var el = document.getElementById(id);
    if (el) sections.push({ id: id, el: el, link: link });
  });
  var debounceTimer;
  window.addEventListener('scroll', function() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function() {
      var current = '';
      for (var i = 0; i < sections.length; i++) {
        if (sections[i].el.getBoundingClientRect().top < 100) current = sections[i].id;
      }
      tocLinks.forEach(function(link) {
        link.classList.toggle('active', link.getAttribute('href') === '#' + current);
      });
    }, 50);
  }, { passive: true });
  tocLinks.forEach(function(link) {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      var id = this.getAttribute('href').slice(1);
      var target = document.getElementById(id);
      if (target) {
        var top = target.getBoundingClientRect().top + window.pageYOffset - 68;
        window.scrollTo({ top: top, behavior: 'smooth' });
        history.pushState(null, '', '#' + id);
      }
    });
  });
  if (window.location.hash) {
    var id = window.location.hash.slice(1);
    var target = document.getElementById(id);
    if (target) {
      setTimeout(function() {
        var top = target.getBoundingClientRect().top + window.pageYOffset - 68;
        window.scrollTo({ top: top, behavior: 'smooth' });
      }, 0);
    }
  }
})();
</script>
</body>
</html>"""

    if map_href != "github_map.html":
        html = html.replace('href="github_map.html"', f'href="{map_href}"')
    Path(output_path).write_text(html)


if __name__ == "__main__":
    main()
