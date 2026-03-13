"""UMAP parameter experiment: compare n_neighbors × min_dist combinations.

Phase 1: Evaluate all 20 configs with embedding-space metrics (no API keys).
Phase 2: Run full Toponymy + audit on a curated set of configs.
"""

import argparse
import copy
import itertools
import time

import numpy as np
import pandas as pd
import umap
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from toponymy import Toponymy, ToponymyClusterer
from toponymy.audit import (
    create_comparison_df,
    create_keyphrase_analysis_df,
    create_layer_summary_df,
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
)

# ── Parameter grid ───────────────────────────────────────────────────────────

N_NEIGHBORS = [5, 15, 30, 50, 100]
MIN_DISTS = [0.0, 0.05, 0.1, 0.25]

PRODUCTION_CONFIG = "nn15_md0.05"

# ── Toponymy defaults (matches 04b_experiment.py) ───────────────────────────

TOPONYMY_DEFAULTS = {
    "min_clusters": 4,
    "object_description": "GitHub repository descriptions",
    "corpus_description": "collection of the top 1,000 most-starred GitHub repositories",
    "exemplar_delimiters": ['    * """', '"""\n'],
    "lowest_detail_level": 0.3,
    "highest_detail_level": 1.0,
}

# ── Spearman sample size ────────────────────────────────────────────────────

SPEARMAN_SAMPLE = 2_000

# ── Stability seeds ─────────────────────────────────────────────────────────

STABILITY_SEEDS = [42, 123, 456, 789, 1024]
STABILITY_K = 15


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


def make_experiment_configs():
    """Cartesian product of n_neighbors × min_dist."""
    configs = []
    for nn, md in itertools.product(N_NEIGHBORS, MIN_DISTS):
        configs.append({
            "name": f"nn{nn}_md{md:.2f}",
            "n_neighbors": nn,
            "min_dist": md,
        })
    return configs


def run_umap(embeddings, config, random_state=42):
    """Fit UMAP with given config, return 2D coords."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=config["n_neighbors"],
        min_dist=config["min_dist"],
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def evaluate_stability(embeddings, config):
    """Run UMAP with multiple seeds and measure k-NN overlap between layouts.

    Returns mean pairwise k-NN stability (fraction of k=STABILITY_K neighbors
    shared between any two runs). Higher = more reproducible layout.
    """
    all_coords = []
    for seed in STABILITY_SEEDS:
        coords = run_umap(embeddings, config, random_state=seed)
        all_coords.append(coords)

    # For each layout, compute k-NN indices
    all_indices = []
    for coords in all_coords:
        nn = NearestNeighbors(n_neighbors=STABILITY_K + 1, metric="euclidean")
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        all_indices.append(indices[:, 1:])  # drop self

    # Pairwise k-NN overlap across all seed pairs
    overlaps = []
    for i, j in itertools.combinations(range(len(STABILITY_SEEDS)), 2):
        per_point = []
        for p in range(len(embeddings)):
            set_i = set(all_indices[i][p])
            set_j = set(all_indices[j][p])
            per_point.append(len(set_i & set_j) / STABILITY_K)
        overlaps.append(np.mean(per_point))

    return np.mean(overlaps)


def precompute_hd_neighbors(embeddings):
    """Pre-compute high-dimensional k-NN indices and Spearman sample distances."""
    print("Pre-computing 512D k-NN (k=50)...")
    nn50 = NearestNeighbors(n_neighbors=51, metric="cosine")
    nn50.fit(embeddings)
    _, hd_indices_50 = nn50.kneighbors(embeddings)
    hd_indices_50 = hd_indices_50[:, 1:]  # drop self

    hd_indices_10 = hd_indices_50[:, :10]

    # Spearman sample
    rng = np.random.RandomState(42)
    n = len(embeddings)
    sample_size = min(SPEARMAN_SAMPLE, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    print(f"Pre-computing 512D pairwise distances for {sample_size} sampled points...")
    hd_dists_sample = pdist(embeddings[sample_idx], metric="cosine")

    return {
        "hd_indices_10": hd_indices_10,
        "hd_indices_50": hd_indices_50,
        "sample_idx": sample_idx,
        "hd_dists_sample": hd_dists_sample,
    }


def evaluate_phase1(embeddings, coords, precomputed):
    """Compute embedding-space metrics for a single UMAP config."""
    n = len(embeddings)

    # Trustworthiness (k=15)
    trust = trustworthiness(embeddings, coords, n_neighbors=15, metric="cosine")

    # k-NN recall
    def knn_recall(hd_indices, k):
        nn_2d = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn_2d.fit(coords)
        _, ld_indices = nn_2d.kneighbors(coords)
        ld_indices = ld_indices[:, 1:]  # drop self
        recalls = []
        for i in range(n):
            hd_set = set(hd_indices[i, :k])
            ld_set = set(ld_indices[i, :k])
            recalls.append(len(hd_set & ld_set) / k)
        return np.mean(recalls)

    recall_10 = knn_recall(precomputed["hd_indices_10"], 10)
    recall_50 = knn_recall(precomputed["hd_indices_50"], 50)

    # Spearman rho (sampled)
    sample_idx = precomputed["sample_idx"]
    ld_dists_sample = pdist(coords[sample_idx], metric="euclidean")
    rho, _ = spearmanr(precomputed["hd_dists_sample"], ld_dists_sample)

    return {
        "trustworthiness_k15": trust,
        "knn_recall_k10": recall_10,
        "knn_recall_k50": recall_50,
        "spearman_rho": rho,
    }


def run_toponymy(coords, embeddings, documents):
    """Fit ToponymyClusterer + Toponymy model on given coords."""
    llm = AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model="claude-sonnet-4-20250514")
    embedder = CohereEmbedder(api_key=CO_API_KEY, model="embed-v4.0")

    clusterer = ToponymyClusterer(min_clusters=TOPONYMY_DEFAULTS["min_clusters"])
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description=TOPONYMY_DEFAULTS["object_description"],
        corpus_description=TOPONYMY_DEFAULTS["corpus_description"],
        exemplar_delimiters=TOPONYMY_DEFAULTS["exemplar_delimiters"],
        lowest_detail_level=TOPONYMY_DEFAULTS["lowest_detail_level"],
        highest_detail_level=TOPONYMY_DEFAULTS["highest_detail_level"],
    )
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=coords,
    )
    return topic_model


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

    # Disambiguation effort
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


def compare_experiments(models, documents):
    """Print pairwise comparison metrics and save markdown comparison."""
    names = list(models.keys())

    print(f"\n{'='*60}")
    print("Audit summaries")
    print(f"{'='*60}")
    for name, model in models.items():
        print_audit_summary(name, model)

    if len(names) < 2:
        print("\nOnly one experiment — skipping pairwise comparison.")
        return

    # Build markdown comparison
    md_lines = ["# UMAP Parameter Experiment — Topic Comparison\n"]

    for layer_idx, label in [(0, "fine"), (-1, "coarse")]:
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

        topic_cols = [f"topic_{n}" for n in names]
        md_lines.append(f"\n## {label.title()} layer\n")

        print(f"\n── {label} layer comparison ──")
        for col in topic_cols:
            n_unique = merged[col].nunique()
            avg_len = merged[col].astype(str).str.len().mean()
            line = f"- **{col}**: {n_unique} unique topics, avg name length {avg_len:.1f} chars"
            md_lines.append(line)
            print(f"  Unique topics ({col}): {n_unique}, avg name length: {avg_len:.1f} chars")

        for name_a, name_b in itertools.combinations(names, 2):
            col_a, col_b = f"topic_{name_a}", f"topic_{name_b}"
            agree = (merged[col_a] == merged[col_b]).mean()
            line = f"- Agreement ({name_a} vs {name_b}): {agree:.1%}"
            md_lines.append(line)
            print(f"  Topic name agreement ({name_a} vs {name_b}): {agree:.1%}")

        # Side-by-side table (first 20 rows)
        md_lines.append(f"\n### Side-by-side topics ({label}, first 20)\n")
        display_cols = ["Cluster ID", "Document Count"] + topic_cols
        table_df = merged[display_cols].head(20)
        md_lines.append(table_df.to_markdown(index=False))

    # Keyphrase overlap (Jaccard) for fine layer
    md_lines.append("\n## Keyphrase overlap (fine, Jaccard)\n")
    print(f"\n── Keyphrase overlap (fine, Jaccard) ──")
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
            line = f"- Jaccard ({name_a} vs {name_b}): {mean_jaccard:.3f}"
            md_lines.append(line)
            print(f"  Mean Jaccard similarity ({name_a} vs {name_b}): {mean_jaccard:.3f}")

    # Save markdown
    comparison_path = EXPERIMENTS_DIR / "umap_comparison.md"
    comparison_path.write_text("\n".join(md_lines) + "\n")
    print(f"\nSaved comparison to {comparison_path}")


def format_name(name):
    """Add * marker for production config."""
    return f"{name} *" if name == PRODUCTION_CONFIG else name


def main():
    parser = argparse.ArgumentParser(description="UMAP parameter experiment")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Run Phase 1 only (no Toponymy, no API keys needed)")
    args = parser.parse_args()

    df, embeddings, documents = load_data()
    print(f"Loaded {len(documents)} documents, embeddings shape {embeddings.shape}")

    configs = make_experiment_configs()

    # ── Phase 1 (skip if results already exist) ─────────────────────────────
    phase1_csv = EXPERIMENTS_DIR / "umap_phase1.csv"
    if phase1_csv.exists():
        print(f"\nPhase 1 results already exist at {phase1_csv}, skipping.")
        phase1_df = pd.read_csv(phase1_csv)
    else:
        print(f"\nPhase 1: evaluating {len(configs)} UMAP configs")

        # Pre-compute HD neighbors
        precomputed = precompute_hd_neighbors(embeddings)

        phase1_rows = []
        for cfg in configs:
            name = cfg["name"]
            print(f"\n  Running UMAP: {name} (n_neighbors={cfg['n_neighbors']}, min_dist={cfg['min_dist']})")

            t0 = time.time()
            coords = run_umap(embeddings, cfg)
            umap_time = time.time() - t0
            print(f"    UMAP fit: {umap_time:.1f}s")

            # Save coords
            exp_dir = EXPERIMENTS_DIR / f"umap_{name}"
            exp_dir.mkdir(exist_ok=True)
            np.savez(exp_dir / "umap_coords.npz", coords=coords)

            # Evaluate
            metrics = evaluate_phase1(embeddings, coords, precomputed)

            # Stability across random seeds
            print(f"    Measuring stability ({len(STABILITY_SEEDS)} seeds)...")
            t1 = time.time()
            stability = evaluate_stability(embeddings, cfg)
            stability_time = time.time() - t1
            metrics["knn_stability_k15"] = stability
            print(f"    trust={metrics['trustworthiness_k15']:.4f}  "
                  f"knn10={metrics['knn_recall_k10']:.4f}  "
                  f"knn50={metrics['knn_recall_k50']:.4f}  "
                  f"spearman={metrics['spearman_rho']:.4f}  "
                  f"stability={stability:.4f} ({stability_time:.1f}s)")

            phase1_rows.append({
                "name": name,
                "n_neighbors": cfg["n_neighbors"],
                "min_dist": cfg["min_dist"],
                **metrics,
                "umap_time_s": round(umap_time, 1),
            })

        phase1_df = pd.DataFrame(phase1_rows).sort_values("trustworthiness_k15", ascending=False)
        phase1_df.to_csv(phase1_csv, index=False)

    # Print ranked table
    print(f"\n{'='*60}")
    print("Phase 1 results (ranked by trustworthiness)")
    print(f"{'='*60}")
    display = phase1_df.copy()
    display["name"] = display["name"].apply(format_name)
    print(display.to_string(index=False, float_format="%.4f"))

    if args.phase1_only:
        print("\n--phase1-only: stopping after Phase 1.")
        return

    # ── Phase 2: curated config selection ────────────────────────────────────
    # We deliberately hand-pick configs rather than auto-selecting top-N by a
    # single metric. Phase 1 showed that trustworthiness and Spearman rho are
    # inversely correlated (local vs global structure tradeoff), and that
    # min_dist=0.0 dominates all local-fidelity metrics by construction (it
    # allows UMAP to pack true neighbors as tightly as possible). Auto-selecting
    # by trustworthiness alone would send only low-n_neighbors, low-min_dist
    # configs to Phase 2, giving no diversity.
    #
    # Selected configs and rationale:
    #   nn5_md0.00  — best trustworthiness overall (0.9298), represents the
    #                 tight-local-neighborhoods extreme
    #   nn15_md0.05 — current production baseline, good middle ground across
    #                 all metrics (trust=0.9101, spearman=0.2573)
    #   nn50_md0.00 — best Spearman (0.2731) among configs with trust > 0.89,
    #                 tests whether better global structure improves topic labels
    #   nn15_md0.25 — highest min_dist at n_neighbors=15, forces points apart
    #                 which may give Toponymy's HDBSCAN clusterer cleaner
    #                 boundaries despite weaker embedding-space metrics
    phase2_names = ["nn5_md0.00", "nn15_md0.05", "nn50_md0.00", "nn15_md0.25"]
    phase2_configs = phase1_df[phase1_df["name"].isin(phase2_names)]
    print(f"\n{'='*60}")
    print(f"Phase 2: running Toponymy on {len(phase2_names)} configs: {', '.join(phase2_names)}")
    print(f"{'='*60}")

    models = {}
    phase2_rows = []
    for _, p1_row in phase2_configs.iterrows():
        name = p1_row["name"]
        exp_dir = EXPERIMENTS_DIR / f"umap_{name}"
        coords = np.load(exp_dir / "umap_coords.npz")["coords"]

        print(f"\n  Running Toponymy for {name}...")
        model = run_toponymy(coords, embeddings, documents)
        models[name] = model

        # Per-config audit CSVs
        layer_summary = create_layer_summary_df(model)
        layer_summary.to_csv(exp_dir / "layer_summary.csv", index=False)

        for layer_idx in [0, -1]:
            label = "fine" if layer_idx == 0 else "coarse"
            kp_df = create_keyphrase_analysis_df(model, layer_index=layer_idx)
            kp_df.to_csv(exp_dir / f"keyphrase_analysis_{label}.csv", index=False)

        for layer_idx in range(len(model.cluster_layers_)):
            comp = create_comparison_df(model, layer_index=layer_idx)
            comp.to_csv(exp_dir / f"comparison_layer{layer_idx}.csv", index=False)

        # Disambiguation stats
        disambig_rows = []
        for li, layer in enumerate(model.cluster_layers_):
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

        # Collect Phase 2 metrics
        finest_layer = model.cluster_layers_[0]
        n_clusters = len(finest_layer.topic_names)
        sizes = np.bincount(finest_layer.cluster_labels[finest_layer.cluster_labels >= 0])
        sizes = sizes[sizes > 0]
        avg_size = sizes.mean() if len(sizes) > 0 else 0

        # Duplicate topic names in finest layer
        names_series = pd.Series(finest_layer.topic_names)
        dup_names = int((names_series.duplicated()).sum())

        # Keyphrase-in-topic rate (fine)
        kp_fine = create_keyphrase_analysis_df(model, layer_index=0)
        kp_rate = kp_fine["keyphrase_in_topic"].mean() if "keyphrase_in_topic" in kp_fine.columns else 0

        # Mean topic name length (fine)
        comp_fine = create_comparison_df(model, layer_index=0)
        mean_name_len = comp_fine["Final LLM Topic Name"].astype(str).str.len().mean()

        phase2_rows.append({
            **p1_row.to_dict(),
            "n_clusters_finest": n_clusters,
            "avg_cluster_size": round(avg_size, 1),
            "duplicate_names": dup_names,
            "keyphrase_in_topic_pct": round(kp_rate * 100, 1),
            "mean_topic_name_len": round(mean_name_len, 1),
        })
        print(f"    clusters={n_clusters}  avg_size={avg_size:.1f}  "
              f"dup_names={dup_names}  kp_rate={kp_rate:.1%}  "
              f"mean_name_len={mean_name_len:.1f}")

    phase2_df = pd.DataFrame(phase2_rows)
    phase2_df.to_csv(EXPERIMENTS_DIR / "umap_phase2.csv", index=False)

    # Print Phase 2 table
    print(f"\n{'='*60}")
    print("Phase 2 results")
    print(f"{'='*60}")
    display2 = phase2_df.copy()
    display2["name"] = display2["name"].apply(format_name)
    print(display2.to_string(index=False, float_format="%.4f"))

    # Cross-config comparison
    compare_experiments(models, documents)

    print("\nDone.")


if __name__ == "__main__":
    main()
