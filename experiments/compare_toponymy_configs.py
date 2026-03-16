"""Experiment framework for comparing Toponymy configurations.

Define experiments as dicts overriding any Toponymy setting (embedder, LLM,
min_clusters, descriptions, delimiters). Runs each config, audits results,
and produces comparison outputs.
"""

import argparse
import copy
import itertools
import tempfile

import anthropic
import nest_asyncio
import numpy as np
import pandas as pd
from toponymy import Toponymy, ToponymyClusterer
from toponymy.audit import (
    create_comparison_df,
    create_keyphrase_analysis_df,
    create_layer_summary_df,
)
from toponymy.embedding_wrappers import CohereEmbedder
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

from pipeline.config import (
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
    {"name": "detail_full_range", "lowest_detail_level": 0.0, "highest_detail_level": 1.0},
    {"name": "detail_medium", "lowest_detail_level": 0.3, "highest_detail_level": 0.8},
    {"name": "detail_concise", "lowest_detail_level": 0.5, "highest_detail_level": 1.0},
    {"name": "detail_broad", "lowest_detail_level": 0.3, "highest_detail_level": 1.0},
]

DEFAULTS = {
    "embedder": CohereEmbedder(api_key=CO_API_KEY, model="embed-v4.0"),
    "llm": AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model="claude-sonnet-4-20250514"),
    "min_clusters": 4,
    "object_description": "GitHub repository descriptions",
    "corpus_description": "collection of the top 10,000 most-starred GitHub repositories",
    "exemplar_delimiters": ['    * """', '"""\n'],
    "lowest_detail_level": 0.0,
    "highest_detail_level": 1.0,
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


def validate_preflight(df, embeddings, coords):
    """Run preflight checks before expensive experiment loop."""
    errors = []

    # Shape consistency
    if embeddings.shape[0] != len(df):
        errors.append(f"embeddings rows ({embeddings.shape[0]}) != df rows ({len(df)})")
    if coords.shape[0] != len(df):
        errors.append(f"coords rows ({coords.shape[0]}) != df rows ({len(df)})")
    if embeddings.shape[1] != 512:
        errors.append(f"embeddings dim ({embeddings.shape[1]}) != 512")

    # API keys
    if not CO_API_KEY:
        errors.append("CO_API_KEY is empty")
    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is empty")

    # Dry-run LLM call
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    except Exception as e:
        errors.append(f"Anthropic API dry-run failed: {e}")

    # Output dir writable
    try:
        with tempfile.NamedTemporaryFile(dir=EXPERIMENTS_DIR, delete=True):
            pass
    except Exception as e:
        errors.append(f"Cannot write to {EXPERIMENTS_DIR}: {e}")

    if errors:
        raise RuntimeError("Preflight checks failed:\n  " + "\n  ".join(errors))

    print("Preflight checks passed.")


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


def save_audit_csvs(model, exp_dir):
    """Save audit CSVs for a fitted model."""
    # Layer summary
    layer_summary = create_layer_summary_df(model)
    layer_summary.to_csv(exp_dir / "audit_layer_summary.csv", index=False)

    # Per-layer comparison and keyphrase CSVs
    n_layers = len(model.cluster_layers_)
    for i in range(n_layers):
        comp = create_comparison_df(model, layer_index=i)
        comp.to_csv(exp_dir / f"audit_comparison_layer{i}.csv", index=False)

        kp = create_keyphrase_analysis_df(model, layer_index=i)
        kp.to_csv(exp_dir / f"audit_keyphrase_layer{i}.csv", index=False)


def run_experiments(df, embeddings, coords, documents, resume=False):
    """Fit Toponymy for each experiment config, return dict of fitted models."""
    default_min = DEFAULTS["min_clusters"]
    base_clusterer = ToponymyClusterer(min_clusters=default_min)
    base_clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    models = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        exp_dir = EXPERIMENTS_DIR / name
        exp_dir.mkdir(exist_ok=True)

        if resume and (exp_dir / "labels.parquet").exists():
            print(f"\n{'=' * 60}")
            print(f"Skipping experiment (resume): {name}")
            print(f"{'=' * 60}")
            continue

        cfg = {**DEFAULTS, **exp}
        print(f"\n{'=' * 60}")
        print(f"Running experiment: {name}")
        print(f"{'=' * 60}")

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
        coarse_labels, fine_labels = extract_labels(topic_model, documents)
        labels_df = pd.DataFrame(
            {
                "full_name": df["full_name"],
                "coarse_label": coarse_labels,
                "fine_label": fine_labels,
            }
        )
        labels_df.to_parquet(exp_dir / "labels.parquet", index=False)
        print(f"  Saved labels to {exp_dir / 'labels.parquet'}")

        save_audit_csvs(topic_model, exp_dir)
        print(f"  Saved audit CSVs to {exp_dir}")

        # Save disambiguation stats
        disambig_rows = []
        for li, layer in enumerate(topic_model.cluster_layers_):
            indices = getattr(layer, "dismbiguation_topic_indices", None)
            if indices is not None:
                disambig_rows.append(
                    {
                        "layer": li,
                        "num_groups": len(indices),
                        "topics_renamed": sum(len(g) for g in indices),
                        "total_topics": len(layer.topic_names),
                    }
                )
        if disambig_rows:
            pd.DataFrame(disambig_rows).to_csv(exp_dir / "disambiguation.csv", index=False)
            print(f"  Saved disambiguation stats to {exp_dir / 'disambiguation.csv'}")

    return models


def compare_experiments():
    """Compare experiments by reading saved CSVs from disk.

    This works whether experiments were just run or skipped via --resume.
    """
    # Discover which experiments have completed (have labels.parquet)
    experiment_names = []
    for exp in EXPERIMENTS:
        exp_dir = EXPERIMENTS_DIR / exp["name"]
        if (exp_dir / "labels.parquet").exists():
            experiment_names.append(exp["name"])

    if not experiment_names:
        print("No completed experiments found. Nothing to compare.")
        return

    print(f"\n{'=' * 60}")
    print("Comparison across experiments")
    print(f"{'=' * 60}")
    print(f"Experiments: {', '.join(experiment_names)}")

    md_lines = ["# Experiment Comparison\n"]

    # Print audit summaries from saved CSVs
    md_lines.append("## Audit Summaries\n")
    for name in experiment_names:
        exp_dir = EXPERIMENTS_DIR / name
        print(f"\n-- {name} --")
        md_lines.append(f"### {name}\n")

        summary_path = exp_dir / "audit_layer_summary.csv"
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            print(summary_df.to_string(index=False))
            md_lines.append("**Layer summary:**\n")
            md_lines.append(summary_df.to_markdown(index=False))
            md_lines.append("")

        # Find all comparison layer files to determine layer count
        layer_idx = 0
        while (exp_dir / f"audit_comparison_layer{layer_idx}.csv").exists():
            comp = pd.read_csv(exp_dir / f"audit_comparison_layer{layer_idx}.csv")
            if "Final LLM Topic Name" in comp.columns:
                lengths = comp["Final LLM Topic Name"].astype(str).str.len()
                print(
                    f"  Layer {layer_idx} avg topic name length: {lengths.mean():.1f} chars "
                    f"(min {lengths.min()}, max {lengths.max()})"
                )
            layer_idx += 1

        # Keyphrase-in-topic rates from saved keyphrase CSVs
        for li in [0]:  # fine layer
            kp_path = exp_dir / f"audit_keyphrase_layer{li}.csv"
            if kp_path.exists():
                kp_df = pd.read_csv(kp_path)
                if "keyphrase_in_topic" in kp_df.columns:
                    rate = kp_df["keyphrase_in_topic"].mean()
                    print(f"  Keyphrase-in-topic rate (layer {li}): {rate:.1%}")
                    md_lines.append(f"Keyphrase-in-topic rate (layer {li}): {rate:.1%}\n")

    if len(experiment_names) < 2:
        print("\nOnly one experiment -- skipping pairwise comparison.")
        md_lines.append("\nOnly one experiment -- no pairwise comparison.\n")
        (EXPERIMENTS_DIR / "comparison_summary.md").write_text("\n".join(md_lines))
        return

    # Build side-by-side comparison CSVs for fine and coarse layers
    md_lines.append("\n## Pairwise Comparisons\n")
    for layer_idx, label in [(0, "fine"), (-1, "coarse")]:
        print(f"\n-- {label} layer comparison --")
        md_lines.append(f"### {label.title()} layer\n")

        rows = []
        for name in experiment_names:
            exp_dir = EXPERIMENTS_DIR / name
            # Resolve negative index: find max layer
            if layer_idx < 0:
                actual_idx = 0
                while (exp_dir / f"audit_comparison_layer{actual_idx + 1}.csv").exists():
                    actual_idx += 1
                idx = actual_idx
            else:
                idx = layer_idx
            comp_path = exp_dir / f"audit_comparison_layer{idx}.csv"
            if comp_path.exists():
                comp = pd.read_csv(comp_path)
                comp = comp.rename(columns={"Final LLM Topic Name": f"topic_{name}"})
                rows.append((name, comp))

        if not rows:
            continue

        # Merge on Cluster ID for side-by-side
        keyphrases_col = "Extracted Keyphrases (Top 5)"
        base_cols = ["Cluster ID", "Document Count"]
        if keyphrases_col in rows[0][1].columns:
            base_cols.append(keyphrases_col)
        merged = rows[0][1][base_cols + [f"topic_{rows[0][0]}"]].copy()
        for exp_name, comp in rows[1:]:
            right = comp[["Cluster ID", f"topic_{exp_name}"]].copy()
            merged = merged.merge(right, on="Cluster ID", how="outer")

        merged.to_csv(EXPERIMENTS_DIR / f"comparison_{label}.csv", index=False)
        print(f"  Saved {EXPERIMENTS_DIR / f'comparison_{label}.csv'}")

        # Summary metrics
        topic_cols = [f"topic_{n}" for n in experiment_names]
        for col in topic_cols:
            if col in merged.columns:
                n_unique = merged[col].nunique()
                avg_len = merged[col].astype(str).str.len().mean()
                line = f"  Unique topics ({col}): {n_unique}, avg name length: {avg_len:.1f} chars"
                print(line)
                md_lines.append(f"- {line.strip()}")

        md_lines.append("")

        # Pairwise agreement
        for name_a, name_b in itertools.combinations(experiment_names, 2):
            col_a, col_b = f"topic_{name_a}", f"topic_{name_b}"
            if col_a in merged.columns and col_b in merged.columns:
                agree = (merged[col_a] == merged[col_b]).mean()
                line = f"  Topic name agreement ({name_a} vs {name_b}): {agree:.1%}"
                print(line)
                md_lines.append(f"- {line.strip()}")

                diff_mask = merged[col_a] != merged[col_b]
                diff_rows = merged[diff_mask].head(3)
                if not diff_rows.empty:
                    print(f"  Example divergent clusters ({name_a} vs {name_b}):")
                    for _, row in diff_rows.iterrows():
                        print(f"    Cluster {row['Cluster ID']}: {row[col_a]!r} vs {row[col_b]!r}")

        md_lines.append("")

    # Keyphrase overlap (Jaccard) for fine layer
    print("\n-- Keyphrase overlap (fine, Jaccard) --")
    md_lines.append("## Keyphrase Overlap (fine layer, Jaccard)\n")
    kp_dfs = {}
    for name in experiment_names:
        kp_path = EXPERIMENTS_DIR / name / "audit_comparison_layer0.csv"
        if kp_path.exists():
            kp = pd.read_csv(kp_path)
            if "Extracted Keyphrases (Top 5)" in kp.columns:
                kp_dfs[name] = kp.set_index("Cluster ID")["Extracted Keyphrases (Top 5)"]

    for name_a, name_b in itertools.combinations(experiment_names, 2):
        if name_a in kp_dfs and name_b in kp_dfs:
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
                line = f"  Mean Jaccard similarity ({name_a} vs {name_b}): {mean_jaccard:.3f}"
                print(line)
                md_lines.append(f"- {line.strip()}")

    # Write summary markdown
    md_lines.append("\n---\n*Generated by 04b_experiment.py*\n")
    (EXPERIMENTS_DIR / "comparison_summary.md").write_text("\n".join(md_lines))
    print(f"\nSaved comparison summary to {EXPERIMENTS_DIR / 'comparison_summary.md'}")


def main():
    parser = argparse.ArgumentParser(description="Run Toponymy detail-level experiments")
    parser.add_argument("--resume", action="store_true", help="Skip experiments with existing labels.parquet")
    args = parser.parse_args()

    df, embeddings, coords, documents = load_data()
    print(f"Loaded {len(documents)} documents")

    validate_preflight(df, embeddings, coords)

    run_experiments(df, embeddings, coords, documents, resume=args.resume)
    compare_experiments()

    print("\nDone.")


if __name__ == "__main__":
    main()
