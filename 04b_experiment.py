"""Experiment framework for comparing Toponymy configurations.

Define experiments as dicts overriding any Toponymy setting (embedder, LLM,
min_clusters, descriptions, delimiters). Runs each config, audits results,
and produces comparison outputs.
"""

import asyncio
import copy

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
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
    EXPERIMENTS_DIR,
    REPOS_PARQUET,
    UMAP_COORDS_NPZ,
)

# ── Experiment configs ───────────────────────────────────────────────────────
# Each dict can override any key from DEFAULTS. Run free/local embedders first.

EXPERIMENTS = [
    {
        "name": "minilm_v2",
        "embedder": SentenceTransformer("all-MiniLM-L6-v2"),
    },
    {
        "name": "cohere_v4",
        "embedder": CohereEmbedder(api_key=CO_API_KEY, model="embed-v4.0"),
    },
]

DEFAULTS = {
    "embedder": CohereEmbedder(api_key=CO_API_KEY, model="embed-v4.0"),
    "llm": AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model="claude-sonnet-4-20250514"),
    "min_clusters": 4,
    "object_description": "GitHub repository descriptions",
    "corpus_description": "collection of the top 1,000 most-starred GitHub repositories",
    "exemplar_delimiters": ['    * """', '"""\n'],
}


def load_data():
    """Load shared data used across all experiments."""
    df = pd.read_parquet(REPOS_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    coords = np.load(UMAP_COORDS_NPZ)["coords"]

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

    return df, embeddings, coords, documents


def extract_labels(model, documents):
    """Extract coarse/fine labels from a fitted Toponymy model."""
    n_layers = len(model.cluster_layers_)
    if n_layers == 0:
        raise ValueError("No cluster layers found")

    coarse_layer = model.cluster_layers_[0]
    fine_layer = model.cluster_layers_[-1]

    coarse_labels = [coarse_layer.topic_name_vector[i] for i in range(len(documents))]
    fine_labels = [fine_layer.topic_name_vector[i] for i in range(len(documents))]
    return coarse_labels, fine_labels


def run_experiments(df, embeddings, coords, documents):
    """Fit Toponymy for each experiment config, return dict of fitted models."""
    default_min = DEFAULTS["min_clusters"]
    base_clusterer = ToponymyClusterer(min_clusters=default_min)
    base_clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    models = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        cfg = {**DEFAULTS, **exp}
        print(f"\n{'='*60}")
        print(f"Running experiment: {name}")
        print(f"{'='*60}")

        # Re-fit clusterer if min_clusters differs from default
        if cfg["min_clusters"] != default_min:
            clusterer = ToponymyClusterer(min_clusters=cfg["min_clusters"])
            clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)
        else:
            clusterer = copy.deepcopy(base_clusterer)

        topic_model = Toponymy(
            llm_wrapper=cfg["llm"],
            text_embedding_model=cfg["embedder"],
            clusterer=clusterer,
            object_description=cfg["object_description"],
            corpus_description=cfg["corpus_description"],
            exemplar_delimiters=cfg["exemplar_delimiters"],
        )
        topic_model.fit(
            objects=documents,
            embedding_vectors=embeddings,
            clusterable_vectors=coords,
        )

        models[name] = topic_model

        # Save per-experiment outputs
        exp_dir = EXPERIMENTS_DIR / name
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

        # Save disambiguation stats (not in audit Excel, only on in-memory model)
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

    return models


def print_audit_summary(name, model):
    """Print audit summary for a single experiment."""
    print(f"\n── {name} ──")
    layer_summary = create_layer_summary_df(model)
    print(layer_summary.to_string(index=False))

    n_layers = len(model.cluster_layers_)
    for layer_idx in range(n_layers):
        label = ["coarse", "mid", "fine"][layer_idx] if n_layers == 3 else f"layer {layer_idx}"
        comp = create_comparison_df(model, layer_index=layer_idx)
        lengths = comp["Final LLM Topic Name"].astype(str).str.len()
        print(f"  Avg topic name length ({label}): {lengths.mean():.1f} chars "
              f"(min {lengths.min()}, max {lengths.max()})")

    # Disambiguation effort: how many topic groups needed renaming per layer
    for layer_idx in range(n_layers):
        label = ["coarse", "mid", "fine"][layer_idx] if n_layers == 3 else f"layer {layer_idx}"
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
        label = "coarse" if layer_idx == 0 else "fine"
        kp_df = create_keyphrase_analysis_df(model, layer_index=layer_idx)
        if "keyphrase_in_topic" in kp_df.columns:
            rate = kp_df["keyphrase_in_topic"].mean()
            print(f"  Keyphrase-in-topic-name rate ({label}): {rate:.1%}")


def compare_experiments(models, documents):
    """Print pairwise comparison metrics and save comparison Excel."""
    names = list(models.keys())

    print(f"\n{'='*60}")
    print("Audit summaries")
    print(f"{'='*60}")
    for name, model in models.items():
        print_audit_summary(name, model)

    if len(names) < 2:
        print("\nOnly one experiment — skipping pairwise comparison.")
        return

    # Build comparison workbook
    with pd.ExcelWriter(EXPERIMENTS_DIR / "comparison.xlsx", engine="openpyxl") as writer:
        for layer_idx, label in [(0, "coarse"), (-1, "fine")]:
            rows = []
            for name, model in models.items():
                comp = create_comparison_df(model, layer_index=layer_idx)
                comp = comp.rename(columns={"Final LLM Topic Name": f"topic_{name}"})
                rows.append((name, comp))

            # Merge on Cluster ID for side-by-side
            keyphrases_col = "Extracted Keyphrases (Top 5)"
            merged = rows[0][1][["Cluster ID", "Document Count", keyphrases_col, f"topic_{rows[0][0]}"]].copy()
            for exp_name, comp in rows[1:]:
                right = comp[["Cluster ID", f"topic_{exp_name}"]].copy()
                merged = merged.merge(right, on="Cluster ID", how="outer")
            merged.to_excel(writer, sheet_name=f"{label}_comparison", index=False)

            # Summary metrics
            topic_cols = [f"topic_{n}" for n in names]
            print(f"\n── {label} layer comparison ──")

            # Topic name agreement
            agree = (merged[topic_cols[0]] == merged[topic_cols[1]]).mean()
            print(f"  Topic name agreement: {agree:.1%}")

            # Unique topic counts and avg name length
            for col in topic_cols:
                n_unique = merged[col].nunique()
                avg_len = merged[col].astype(str).str.len().mean()
                print(f"  Unique topics ({col}): {n_unique}, avg name length: {avg_len:.1f} chars")

            # Show divergent clusters
            diff_mask = merged[topic_cols[0]] != merged[topic_cols[1]]
            diff_rows = merged[diff_mask].head(5)
            if not diff_rows.empty:
                print(f"  Example divergent clusters ({label}):")
                for _, row in diff_rows.iterrows():
                    print(f"    Cluster {row['Cluster ID']}: "
                          f"{row[topic_cols[0]]!r} vs {row[topic_cols[1]]!r}")

        # Keyphrase overlap (Jaccard) for coarse layer
        print(f"\n── Keyphrase overlap (coarse, Jaccard) ──")
        kp_dfs = {}
        for name, model in models.items():
            kp = create_comparison_df(model, layer_index=0)
            kp_dfs[name] = kp.set_index("Cluster ID")["Extracted Keyphrases (Top 5)"]

        common_ids = kp_dfs[names[0]].index.intersection(kp_dfs[names[1]].index)
        jaccard_scores = []
        for cid in common_ids:
            set_a = set(str(kp_dfs[names[0]].get(cid, "")).split(", "))
            set_b = set(str(kp_dfs[names[1]].get(cid, "")).split(", "))
            if set_a or set_b:
                jaccard = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 0
                jaccard_scores.append(jaccard)

        if jaccard_scores:
            mean_jaccard = np.mean(jaccard_scores)
            print(f"  Mean Jaccard similarity: {mean_jaccard:.3f}")

    print(f"\nSaved comparison to {EXPERIMENTS_DIR / 'comparison.xlsx'}")


def main():
    df, embeddings, coords, documents = load_data()
    print(f"Loaded {len(documents)} documents")

    models = run_experiments(df, embeddings, coords, documents)
    compare_experiments(models, documents)

    print("\nDone.")


if __name__ == "__main__":
    main()
