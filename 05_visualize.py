"""Generate interactive DataMapPlot visualization."""

import base64
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import datamapplot
import glasbey
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datamapplot.edge_bundling import bundle_edges
from datamapplot.interactive_rendering import compute_percentile_bounds
from matplotlib.collections import LineCollection
from sklearn.neighbors import NearestNeighbors

from config import (
    DOCS_INDEX_HTML,
    EMBEDDINGS_NPZ,
    GITHUB_MAP_HTML,
    LABELS_PARQUET,
    METHODOLOGY_HTML,
    REPOS_PARQUET,
    UMAP_COORDS_NPZ,
)

# docs/methodology.html is the hand-authored source for the methodology page.
# _write_methodology() reads it and writes an adjusted copy to data/.
METHODOLOGY_SOURCE_HTML = Path(__file__).parent / "docs" / "methodology.html"

FILTER_PANEL_HTML = Path(__file__).parent / "filter_panel.html"


def _build_edge_bundle(coords, embeddings):
    """Build KNN edge bundle from embeddings and return background_image kwargs."""
    print("Building edge bundle...")
    n = len(coords)

    # 1. Build KNN graph in 512D embedding space
    nn = NearestNeighbors(n_neighbors=15, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(embeddings)
    _, nn_indices = nn.kneighbors(embeddings)

    source = np.repeat(np.arange(n), 15)
    target = nn_indices.flatten()
    mask = source != target
    edges = pd.DataFrame({"source": source[mask], "target": target[mask]})

    # 2. Transform coordinates to DataMapPlot's internal space (centered, scale=30)
    raw_bounds = compute_percentile_bounds(coords)
    raw_w = raw_bounds[1] - raw_bounds[0]
    raw_h = raw_bounds[3] - raw_bounds[2]
    raw_scale = max(raw_w, raw_h)
    coords_center = coords.mean(axis=0)
    dmp_scale = 30.0 / raw_scale
    transformed_coords = dmp_scale * (coords - coords_center)

    # 3. Bundle edges
    color_list = ["#888888"] * n
    segments, seg_colors = bundle_edges(transformed_coords, color_list, edges=edges)

    # 4. Compute image bounds with 5% padding
    pad_frac = 0.05
    tc = transformed_coords
    x_range = tc[:, 0].max() - tc[:, 0].min()
    y_range = tc[:, 1].max() - tc[:, 1].min()
    bounds = [
        float(tc[:, 0].min() - pad_frac * x_range),
        float(tc[:, 1].min() - pad_frac * y_range),
        float(tc[:, 0].max() + pad_frac * x_range),
        float(tc[:, 1].max() + pad_frac * y_range),
    ]

    # 5. Render to transparent PNG
    fig_w = 20
    fig_h = fig_w * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    lines = np.stack([segments[:, :2], segments[:, 2:]], axis=1)
    lc = LineCollection(lines, colors=seg_colors, linewidths=0.3, alpha=0.3)
    ax.add_collection(lc)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=100)
    plt.close(fig)
    buf.seek(0)
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    print("Edge bundle complete.")
    return {"background_image": data_uri, "background_image_bounds": bounds}

# Custom JS injected at render time: hide boundaries immediately after async
# creation, and patch pointLayer.clone to preserve radiusMinPixels/radiusMaxPixels.
CUSTOM_JS = """
// Hide edge-bundle image layer initially (synchronous — exists at custom_js time)
if (datamap.imageLayer) {
    var idx = datamap.layers.indexOf(datamap.imageLayer);
    datamap.imageLayer = datamap.imageLayer.clone({visible: false});
    datamap.layers[idx] = datamap.imageLayer;
    datamap.deckgl.setProps({layers: [].concat(datamap.layers)});
}

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


def _build_point_labels_js():
    """Return JS IIFE that adds per-repo text labels visible on zoom."""
    return """\
(function() {
  window.addEventListener('datamapReady', function(e) {
    var datamap = e.detail.datamap;
    var hoverData = e.detail.hoverData;
    if (!datamap.pointLayer) return;
    var positions = datamap.pointLayer.props.data.attributes.getPosition.value;
    var titles = hoverData.project_title || hoverData.hover_text;
    var n = titles.length;
    var allData = [];
    for (var i = 0; i < n; i++) {
      var t = titles[i];
      if (!t) continue;
      allData.push({ text: t, position: [positions[i * 2], positions[i * 2 + 1]], idx: i });
    }
    var visibleData = allData;
    var charSet = new Set();
    for (var i = 0; i < allData.length; i++) {
      var text = allData[i].text;
      for (var j = 0; j < text.length; j++) charSet.add(text[j]);
    }
    var characterSet = Array.from(charSet);
    var initialZoom = datamap.deckgl.props.initialViewState.zoom;
    var fadeStart = initialZoom + 4;
    var fadeEnd = initialZoom + 6;
    var lastStep = -1;
    var enabled = true;
    function buildLayer(opacity) {
      return new deck.TextLayer({
        id: 'pointLabelLayer',
        data: visibleData,
        getText: function(d) { return d.text; },
        getPosition: function(d) { return d.position; },
        getSize: 0.08,
        sizeMinPixels: 0,
        sizeMaxPixels: 14,
        sizeUnits: 'common',
        getColor: [0, 0, 0],
        fontWeight: 400,
        fontFamily: 'Roboto, Arial, sans-serif',
        characterSet: characterSet,
        background: false,
        wordBreak: 'break-word',
        maxWidth: 24,
        getPixelOffset: [0, 10],
        opacity: opacity,
        parameters: { depthTest: false }
      });
    }
    var dpIdx = datamap.layers.findIndex(function(l) { return l.id === 'dataPointLayer'; });
    if (dpIdx >= 0) {
      datamap.layers.splice(dpIdx + 1, 0, buildLayer(0));
    } else {
      datamap.layers.push(buildLayer(0));
    }
    datamap.deckgl.setProps({ layers: [].concat(datamap.layers) });
    function updateOpacity(opacity) {
      var idx = datamap.layers.findIndex(function(l) { return l.id === 'pointLabelLayer'; });
      if (idx === -1) return;
      datamap.layers[idx] = buildLayer(opacity);
      datamap.deckgl.setProps({ layers: [].concat(datamap.layers) });
    }
    function currentOpacity() {
      var vs = datamap.deckgl.props.viewState || datamap.deckgl.props.initialViewState;
      var zoom = vs ? vs.zoom : initialZoom;
      var t = Math.min(1, Math.max(0, (zoom - fadeStart) / (fadeEnd - fadeStart)));
      return Math.round(t * 20) / 20;
    }
    datamap._pointLabels = {
      setEnabled: function(v) {
        enabled = v;
        if (!v) { lastStep = -1; updateOpacity(0); }
        else { var op = currentOpacity(); lastStep = Math.round(op * 20); updateOpacity(op); }
      },
      updateVisibility: function(selectedSet) {
        if (!selectedSet) { visibleData = allData; }
        else { visibleData = allData.filter(function(d) { return selectedSet.has(d.idx); }); }
        if (enabled) updateOpacity(currentOpacity());
      }
    };
    var origHandler = datamap.deckgl.props.onViewStateChange || null;
    datamap.deckgl.setProps({
      onViewStateChange: function(params) {
        var result;
        if (origHandler) result = origHandler(params);
        if (!enabled) return result;
        var zoom = params.viewState.zoom;
        var t = Math.min(1, Math.max(0, (zoom - fadeStart) / (fadeEnd - fadeStart)));
        var step = Math.round(t * 20);
        if (step !== lastStep) {
          lastStep = step;
          var opacity = step / 20;
          requestAnimationFrame(function() { updateOpacity(opacity); });
        }
        return result;
      }
    });
  });
})();"""


def main():
    # Load data
    df = pd.read_parquet(REPOS_PARQUET)
    labels_df = pd.read_parquet(LABELS_PARQUET)
    coords = np.load(UMAP_COORDS_NPZ)["coords"]
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]

    # Build edge bundle background image
    edge_bundle_kwargs = _build_edge_bundle(coords, embeddings)

    # Collect all label layers (label_layer_0 = coarsest, label_layer_1, ... = finer)
    label_columns = sorted(c for c in labels_df.columns if c.startswith("label_layer_"))
    topic_name_vectors = [labels_df[c].values for c in label_columns]

    # ── Hover text ───────────────────────────────────────────────────────────
    has_forks = "fork_count" in df.columns
    has_summary = "summary" in df.columns
    has_title = "project_title" in df.columns

    # Build per-row metadata fields for hover card
    hover_stars = [f"{row['stargazers_count']:,}" for _, row in df.iterrows()]
    hover_forks = [f"{row['fork_count']:,}" if has_forks else "" for _, row in df.iterrows()]
    hover_langs = [(row["language"] or "") for _, row in df.iterrows()]

    hover_text = df["full_name"].tolist()  # required by DataMapPlot

    hover_text_html_template = (
        '<div class="hc">'
        '  <div class="hc-header">'
        '    <div class="hc-title">{project_title}</div>'
        '    <div class="hc-repo">{full_name}</div>'
        '  </div>'
        '  <div class="hc-tagline">{tagline}</div>'
        '  <div class="hc-stats">'
        '    <span class="hc-chip">★ {hover_stars}</span>'
        '    <span class="hc-chip">⑂ {hover_forks}</span>'
        '    <span class="hc-chip hc-lang">{hover_lang}</span>'
        '  </div>'
        '  <div class="hc-type">{project_type}</div>'
        '  <div class="hc-summary">{summary}</div>'
        '</div>'
    )

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

    # 3. License Family (categorical, grouped from license)
    licenses = df["license"].fillna("").replace("", "None").values
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

    # 4. Project Type (categorical, from LLM extraction)
    has_project_type = "project_type" in df.columns
    if has_project_type:
        project_types = df["project_type"].fillna("Other").replace("", "Other").values
        unique_types = sorted(set(project_types))
        type_palette = glasbey.create_palette(palette_size=len(unique_types))
        type_color_mapping = dict(zip(unique_types, type_palette))

    # 5. Repo Created Date (datetime, for continuous colormap)
    now = datetime.now(tz=timezone.utc)
    created_dates = pd.to_datetime(df["created_at"], utc=True).values

    # 6–9. Compute data for extra colormaps (guarded for backward compat)
    if "owner_type" in df.columns:
        owner_types = df["owner_type"].fillna("Unknown").replace("", "Unknown").values
        unique_owners = sorted(set(owner_types))
        owner_palette = glasbey.create_palette(palette_size=len(unique_owners))
        owner_color_mapping = dict(zip(unique_owners, owner_palette))

    if "pushed_at" in df.columns:
        days_since_push = np.array([
            (now - pd.to_datetime(d, utc=True).to_pydatetime()).days
            if d else 9999
            for d in df["pushed_at"]
        ], dtype=float)
    else:
        days_since_push = np.zeros(len(df), dtype=float)

    if "fork_count" in df.columns:
        fork_counts_log = np.log10(df["fork_count"].values.astype(float).clip(min=1))

    if "open_issue_count" in df.columns:
        open_issues_log = np.log10(df["open_issue_count"].values.astype(float).clip(min=1))

    # ── Build the interactive plot ───────────────────────────────────────────

    # Extra point data for filters (all as strings for DataMapPlot serialization)
    created_years = pd.to_datetime(df["created_at"], utc=True).dt.year.values
    summaries = df["summary"].fillna("").values if has_summary else [""] * len(df)
    taglines = df["tagline"].fillna("").values if "tagline" in df.columns else [""] * len(df)
    project_titles = df["project_title"].fillna("").values if has_title else df["full_name"].str.split("/").str[1].values
    search_text = [
        f"{fn} {title} {lang or ''} {tag} {summ}"
        for fn, title, lang, tag, summ in zip(df["full_name"], project_titles, df["language"].fillna(""), taglines, summaries)
    ]

    project_type_values = project_types if has_project_type else ["Other"] * len(df)
    extra_data = pd.DataFrame({
        "full_name": df["full_name"].values,
        "project_title": project_titles,
        "project_type": project_type_values,
        "tagline": taglines,
        "summary": summaries,
        "hover_stars": hover_stars,
        "hover_forks": hover_forks,
        "hover_lang": hover_langs,
        "stars": df["stargazers_count"].astype(str).values,
        "created_year": created_years.astype(str),
        "days_since_push": days_since_push.astype(int).astype(str) if "pushed_at" in df.columns else "0",
        "forks": df["fork_count"].astype(str).values if "fork_count" in df.columns else "0",
        "open_issues": df["open_issue_count"].astype(str).values if "open_issue_count" in df.columns else "0",
        "primary_language": languages,
        "search_text": search_text,
    })

    # Order: categoricals first, then temporal, then count metrics
    all_rawdata = [languages, license_families]
    all_metadata = [
            {
                "field": "language",
                "description": "Primary Language",
                "kind": "categorical",
                "color_mapping": lang_color_mapping,
            },
            {
                "field": "license_family",
                "description": "License Family",
                "kind": "categorical",
                "color_mapping": family_color_mapping,
            },
    ]

    # Project Type (categorical)
    if has_project_type:
        all_rawdata.append(project_types)
        all_metadata.append({
            "field": "project_type",
            "description": "Project Type",
            "kind": "categorical",
            "color_mapping": type_color_mapping,
        })

    # Owner Type (categorical)
    if "owner_type" in df.columns:
        all_rawdata.append(owner_types)
        all_metadata.append({
            "field": "owner_type",
            "description": "Owner Type",
            "kind": "categorical",
            "color_mapping": owner_color_mapping,
        })

    # Created Date (temporal)
    all_rawdata.append(created_dates)
    all_metadata.append({
            "field": "created",
            "description": "Created Date",
            "kind": "datetime",
            "cmap": "viridis",
    })

    # Last Push days (temporal/continuous)
    if "pushed_at" in df.columns:
        all_rawdata.append(np.log10(days_since_push.clip(min=1)))
        all_metadata.append({
            "field": "last_push",
            "description": "Last Push days (log10)",
            "kind": "continuous",
            "cmap": "RdYlGn_r",
        })

    # Star Count (continuous)
    all_rawdata.append(star_counts)
    all_metadata.append({
            "field": "stars",
            "description": "Star Count (log10)",
            "kind": "continuous",
            "cmap": "YlOrRd",
    })

    # Fork Count (continuous)
    if "fork_count" in df.columns:
        all_rawdata.append(fork_counts_log)
        all_metadata.append({
            "field": "forks",
            "description": "Fork Count (log10)",
            "kind": "continuous",
            "cmap": "YlGnBu",
        })

    # Open Issues (continuous)
    if "open_issue_count" in df.columns:
        all_rawdata.append(open_issues_log)
        all_metadata.append({
            "field": "open_issues",
            "description": "Open Issues (log10)",
            "kind": "continuous",
            "cmap": "BuPu",
        })

    tooltip_css = """
        font-family: 'DM Sans', system-ui, sans-serif;
        font-size: 13px;
        font-weight: 400;
        color: #1a1a2e !important;
        background: linear-gradient(135deg, #ffffffe8, #f8f9fee8) !important;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 10px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04);
        max-width: 340px;
        padding: 0 !important;
        overflow: hidden;
    """

    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600&family=JetBrains+Mono:wght@400;500&display=swap');

    .hc {
        padding: 14px 16px 12px;
    }
    .hc-header {
        margin-bottom: 10px;
    }
    .hc-title {
        font-weight: 600;
        font-size: 15px;
        color: #0d1117;
        line-height: 1.3;
        letter-spacing: -0.01em;
    }
    .hc-repo {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        color: #57606a;
        margin-top: 2px;
        letter-spacing: -0.02em;
    }
    .hc-tagline {
        font-size: 13px;
        font-style: italic;
        font-weight: 400;
        color: #656d76;
        line-height: 1.4;
        margin-bottom: 10px;
    }
    .hc-tagline:empty {
        display: none;
    }
    .hc-stats {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-bottom: 10px;
    }
    .hc-chip {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(0, 0, 0, 0.04);
        color: #424a53;
        white-space: nowrap;
    }
    .hc-chip:empty {
        display: none;
    }
    .hc-lang {
        background: transparent;
        border: 1px solid rgba(0, 0, 0, 0.15);
        color: #57606a;
    }
    .hc-type {
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8b6cc1;
        margin-bottom: 8px;
    }
    .hc-type:empty {
        display: none;
    }
    .hc-summary {
        font-size: 12.5px;
        line-height: 1.5;
        color: #3d4752;
        display: -webkit-box;
        -webkit-line-clamp: 10;
        -webkit-box-orient: vertical;
        overflow: hidden;
        border-top: 1px solid rgba(0, 0, 0, 0.06);
        padding-top: 8px;
    }
    .hc-summary:empty {
        display: none;
    }
    """

    fig = datamapplot.create_interactive_plot(
        coords,
        *topic_name_vectors,
        hover_text=hover_text,
        hover_text_html_template=hover_text_html_template,
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
        custom_css=custom_css,
        tooltip_css=tooltip_css,
        cluster_boundary_polygons=True,
        cluster_boundary_line_width=1.0,
        darkmode=False,
        **edge_bundle_kwargs,
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
        "height:calc(100vh - 44px);z-index:1;box-sizing:border-box;padding:0",
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
    point_labels_js = "<script>" + _build_point_labels_js() + "</script>"
    html = html.replace("</html>", js_section + "\n" + point_labels_js + "\n</html>", 1)

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
.color-swatch{min-width:60px;}
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


def _write_methodology(output_path):
    """Write data/ copy of methodology page with nav links adjusted for local use.

    The source of truth is docs/methodology.html (uses href="index.html" for GitHub Pages).
    This function reads that source and writes a copy with href="github_map.html" for data/.
    """
    html = METHODOLOGY_SOURCE_HTML.read_text()
    html = html.replace('href="index.html"', 'href="github_map.html"')
    Path(output_path).write_text(html)


if __name__ == "__main__":
    main()
