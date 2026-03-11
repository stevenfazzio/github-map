"""Experiment framework for comparing PCA pre-reduction before UMAP.

Varies PCA dimensionality (None, 50, 125, 250) before UMAP 2D reduction,
then runs Toponymy with identical params to measure how pre-reduction
affects cluster structure and topic labels.
"""

import asyncio
import copy
import itertools

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from toponymy import Toponymy, ToponymyClusterer
from toponymy.audit import (
    create_comparison_df,
    create_keyphrase_analysis_df,
    create_layer_summary_df,
    export_audit_excel,
)
from toponymy.embedding_wrappers import CohereEmbedder
from toponymy.llm_wrappers import AsyncAnthropicNamer

import nest_asyncio

nest_asyncio.apply()

from config import (
    ANTHROPIC_API_KEY,
    CO_API_KEY,
    EMBEDDINGS_NPZ,
    PCA_EXPERIMENTS_DIR,
    REPOS_PARQUET,
)

# ── Experiment configs ───────────────────────────────────────────────────────
# Each dict specifies pca_dims (None = no PCA, use raw 512d embeddings).

EXPERIMENTS = [
    {
        "name": "baseline_no_pca",
        "pca_dims": None,
    },
    {
        "name": "pca_250",
        "pca_dims": 250,
    },
    {
        "name": "pca_125",
        "pca_dims": 125,
    },
    {
        "name": "pca_50",
        "pca_dims": 50,
    },
]

DEFAULTS = {
    "embedder": CohereEmbedder(api_key=CO_API_KEY, model="embed-v4.0"),
    "llm": AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model="claude-sonnet-4-20250514"),
    "min_clusters": 4,
    "object_description": "GitHub repository descriptions",
    "corpus_description": "collection of the top 1,000 most-starred GitHub repositories",
    "exemplar_delimiters": ['    * """', '"""\n'],
    "lowest_detail_level": 0.5,
    "highest_detail_level": 1.0,
}


def reduce_embeddings(embeddings, pca_dims):
    """Optionally apply PCA then UMAP to produce 2D coordinates.

    Returns (coords, pca_info) where pca_info is a dict with variance stats
    (or None if no PCA was applied).
    """
    if pca_dims is not None:
        pca = PCA(n_components=pca_dims, random_state=42)
        reduced = pca.fit_transform(embeddings)
        cumulative_var = pca.explained_variance_ratio_.cumsum()
        pca_info = {
            "n_components": pca_dims,
            "explained_variance_total": float(cumulative_var[-1]),
            "top_10_variance": float(cumulative_var[min(9, len(cumulative_var) - 1)]),
        }
        print(f"  PCA {embeddings.shape[1]}d → {pca_dims}d: "
              f"{pca_info['explained_variance_total']:.1%} variance retained")
        umap_input = reduced
    else:
        pca_info = None
        umap_input = embeddings
        print(f"  No PCA: using raw {embeddings.shape[1]}d embeddings")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(umap_input)
    print(f"  UMAP → {coords.shape}")
    return coords, pca_info


def load_data():
    """Load shared data used across all experiments."""
    df = pd.read_parquet(REPOS_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]

    MAX_README_CHARS = 2_000
    has_summary = "summary" in df.columns

    documents = []
    for _, row in df.iterrows():
        text = ""
        if has_summary and isinstance(row.get("summary"), str):
            text = row["summary"].strip()
        if not text:
            text = row["readme"].strip() if isinstance(row["readme"], str) else ""
            text = text[:MAX_README_CHARS]
        if not text:
            text = row["description"].strip() if isinstance(row["description"], str) else ""
        if not text:
            text = row["full_name"]
        documents.append(text)

    return df, embeddings, documents


def extract_labels(model, documents):
    """Extract coarse/fine labels from a fitted Toponymy model."""
    n_layers = len(model.cluster_layers_)
    if n_layers == 0:
        raise ValueError("No cluster layers found")

    coarse_layer = model.cluster_layers_[-1]
    fine_layer = model.cluster_layers_[0]

    coarse_labels = [coarse_layer.topic_name_vector[i] for i in range(len(documents))]
    fine_labels = [fine_layer.topic_name_vector[i] for i in range(len(documents))]
    return coarse_labels, fine_labels


def run_experiments(df, embeddings, documents):
    """Run PCA+UMAP+Toponymy for each experiment config."""
    models = {}
    all_pca_info = {}

    for exp in EXPERIMENTS:
        name = exp["name"]
        pca_dims = exp["pca_dims"]
        cfg = {**DEFAULTS, **exp}
        print(f"\n{'='*60}")
        print(f"Running experiment: {name} (PCA dims: {pca_dims or 'None'})")
        print(f"{'='*60}")

        # Step 1: PCA + UMAP
        coords, pca_info = reduce_embeddings(embeddings, pca_dims)
        all_pca_info[name] = pca_info

        # Step 2: Cluster on 2D coords, but use original 512d embeddings for exemplars
        clusterer = ToponymyClusterer(min_clusters=cfg["min_clusters"])
        clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

        # Log cluster counts per layer
        for li, layer in enumerate(clusterer.cluster_layers_):
            n_clusters = len(set(layer.labels_)) - (1 if -1 in layer.labels_ else 0)
            print(f"  Layer {li}: {n_clusters} clusters")

        # Step 3: Fit Toponymy
        topic_model = Toponymy(
            llm_wrapper=cfg["llm"],
            text_embedding_model=cfg["embedder"],
            clusterer=clusterer,
            object_description=cfg["object_description"],
            corpus_description=cfg["corpus_description"],
            exemplar_delimiters=cfg["exemplar_delimiters"],
            lowest_detail_level=cfg["lowest_detail_level"],
            highest_detail_level=cfg["highest_detail_level"],
        )
        topic_model.fit(
            objects=documents,
            embedding_vectors=embeddings,
            clusterable_vectors=coords,
        )

        models[name] = topic_model

        # Save per-experiment outputs
        exp_dir = PCA_EXPERIMENTS_DIR / name
        exp_dir.mkdir(exist_ok=True)

        coarse_labels, fine_labels = extract_labels(topic_model, documents)
        labels_df = pd.DataFrame({
            "full_name": df["full_name"],
            "coarse_label": coarse_labels,
            "fine_label": fine_labels,
        })
        labels_df.to_parquet(exp_dir / "labels.parquet", index=False)
        print(f"  Saved labels to {exp_dir / 'labels.parquet'}")

        export_audit_excel(topic_model, filename=str(exp_dir / "audit.xlsx"))
        print(f"  Saved audit to {exp_dir / 'audit.xlsx'}")

        # Save PCA info
        if pca_info:
            pd.DataFrame([pca_info]).to_csv(exp_dir / "pca_info.csv", index=False)

        # Save disambiguation stats
        disambig_rows = []
        for li, layer in enumerate(topic_model.cluster_layers_):
            indices = getattr(layer, "dismbiguation_topic_indices", None)
            if indices is not None:
                disambig_rows.append({
                    "layer": li,
                    "num_groups": len(indices),
                    "topics_renamed": sum(len(g) for g in indices),
                    "total_topics": len(layer.topic_names),
                })
        if disambig_rows:
            pd.DataFrame(disambig_rows).to_csv(exp_dir / "disambiguation.csv", index=False)
            print(f"  Saved disambiguation stats to {exp_dir / 'disambiguation.csv'}")

    return models, all_pca_info


def print_audit_summary(name, model):
    """Print audit summary for a single experiment."""
    print(f"\n── {name} ──")
    layer_summary = create_layer_summary_df(model)
    print(layer_summary.to_string(index=False))

    n_layers = len(model.cluster_layers_)
    for layer_idx in range(n_layers):
        label = ["fine", "mid", "coarse"][layer_idx] if n_layers == 3 else f"layer {layer_idx}"
        comp = create_comparison_df(model, layer_index=layer_idx)
        lengths = comp["Final LLM Topic Name"].astype(str).str.len()
        print(f"  Avg topic name length ({label}): {lengths.mean():.1f} chars "
              f"(min {lengths.min()}, max {lengths.max()})")

    for layer_idx in range(n_layers):
        label = ["fine", "mid", "coarse"][layer_idx] if n_layers == 3 else f"layer {layer_idx}"
        layer = model.cluster_layers_[layer_idx]
        indices = getattr(layer, "dismbiguation_topic_indices", None)
        if indices is not None:
            n_groups = len(indices)
            n_topics = sum(len(g) for g in indices)
            total = len(layer.topic_names)
            print(f"  Disambiguation ({label}): {n_groups} groups, "
                  f"{n_topics}/{total} topics needed renaming")
        else:
            print(f"  Disambiguation ({label}): no data (attribute not found)")

    for layer_idx in [0, -1]:
        label = "fine" if layer_idx == 0 else "coarse"
        kp_df = create_keyphrase_analysis_df(model, layer_index=layer_idx)
        if "keyphrase_in_topic" in kp_df.columns:
            rate = kp_df["keyphrase_in_topic"].mean()
            print(f"  Keyphrase-in-topic-name rate ({label}): {rate:.1%}")


def compare_experiments(models, documents, all_pca_info):
    """Print pairwise comparison metrics and save comparison Excel."""
    names = list(models.keys())

    # Print PCA variance summary
    print(f"\n{'='*60}")
    print("PCA variance summary")
    print(f"{'='*60}")
    for name, info in all_pca_info.items():
        if info:
            print(f"  {name}: {info['n_components']} dims, "
                  f"{info['explained_variance_total']:.1%} variance retained")
        else:
            print(f"  {name}: no PCA (raw 512d)")

    # Print cluster count comparison
    print(f"\n{'='*60}")
    print("Cluster counts per experiment")
    print(f"{'='*60}")
    for name, model in models.items():
        counts = []
        for li, layer in enumerate(model.cluster_layers_):
            n_clusters = len(set(layer.labels_)) - (1 if -1 in layer.labels_ else 0)
            counts.append(f"layer {li}: {n_clusters}")
        print(f"  {name}: {', '.join(counts)}")

    print(f"\n{'='*60}")
    print("Audit summaries")
    print(f"{'='*60}")
    for name, model in models.items():
        print_audit_summary(name, model)

    if len(names) < 2:
        print("\nOnly one experiment — skipping pairwise comparison.")
        return

    # Build comparison workbook
    with pd.ExcelWriter(PCA_EXPERIMENTS_DIR / "comparison.xlsx", engine="openpyxl") as writer:
        for layer_idx, label in [(0, "fine"), (-1, "coarse")]:
            rows = []
            for name, model in models.items():
                comp = create_comparison_df(model, layer_index=layer_idx)
                comp = comp.rename(columns={"Final LLM Topic Name": f"topic_{name}"})
                rows.append((name, comp))

            keyphrases_col = "Extracted Keyphrases (Top 5)"
            merged = rows[0][1][["Cluster ID", "Document Count", keyphrases_col, f"topic_{rows[0][0]}"]].copy()
            for exp_name, comp in rows[1:]:
                right = comp[["Cluster ID", f"topic_{exp_name}"]].copy()
                merged = merged.merge(right, on="Cluster ID", how="outer")
            merged.to_excel(writer, sheet_name=f"{label}_comparison", index=False)

            # Summary metrics
            topic_cols = [f"topic_{n}" for n in names]
            print(f"\n── {label} layer comparison ──")

            for col in topic_cols:
                n_unique = merged[col].nunique()
                avg_len = merged[col].astype(str).str.len().mean()
                print(f"  Unique topics ({col}): {n_unique}, avg name length: {avg_len:.1f} chars")

            for name_a, name_b in itertools.combinations(names, 2):
                col_a, col_b = f"topic_{name_a}", f"topic_{name_b}"
                agree = (merged[col_a] == merged[col_b]).mean()
                print(f"  Topic name agreement ({name_a} vs {name_b}): {agree:.1%}")

                diff_mask = merged[col_a] != merged[col_b]
                diff_rows = merged[diff_mask].head(5)
                if not diff_rows.empty:
                    print(f"  Example divergent clusters ({name_a} vs {name_b}):")
                    for _, row in diff_rows.iterrows():
                        print(f"    Cluster {row['Cluster ID']}: "
                              f"{row[col_a]!r} vs {row[col_b]!r}")

        # Keyphrase overlap (Jaccard) for coarse layer
        print(f"\n── Keyphrase overlap (coarse, Jaccard) ──")
        kp_dfs = {}
        for name, model in models.items():
            kp = create_comparison_df(model, layer_index=0)
            kp_dfs[name] = kp.set_index("Cluster ID")["Extracted Keyphrases (Top 5)"]

        for name_a, name_b in itertools.combinations(names, 2):
            common_ids = kp_dfs[name_a].index.intersection(kp_dfs[name_b].index)
            jaccard_scores = []
            for cid in common_ids:
                set_a = set(str(kp_dfs[name_a].get(cid, "")).split(", "))
                set_b = set(str(kp_dfs[name_b].get(cid, "")).split(", "))
                if set_a or set_b:
                    jaccard = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 0
                    jaccard_scores.append(jaccard)

            if jaccard_scores:
                mean_jaccard = np.mean(jaccard_scores)
                print(f"  Mean Jaccard similarity ({name_a} vs {name_b}): {mean_jaccard:.3f}")

    print(f"\nSaved comparison to {PCA_EXPERIMENTS_DIR / 'comparison.xlsx'}")


def main():
    df, embeddings, documents = load_data()
    print(f"Loaded {len(documents)} documents, embeddings shape: {embeddings.shape}")

    models, all_pca_info = run_experiments(df, embeddings, documents)
    compare_experiments(models, documents, all_pca_info)

    print("\nDone.")


if __name__ == "__main__":
    main()
