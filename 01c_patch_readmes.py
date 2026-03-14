"""Backfill missing READMEs for repos that use alternate filenames.

01_fetch_repos.py originally only checked README.md/readme.md. This script
fetches READMEs for repos with empty/short READMEs using all 11 filename
variants, then re-selects repos.parquet with the recovered repos included.

Safe to run multiple times — exits early if no targets remain.
"""

import json
import os
import shutil
import tempfile

import pandas as pd
from tqdm import tqdm

from config import (
    DATA_DIR,
    FETCH_OVERSHOOT_COUNT,
    GRAPHQL_BATCH_SIZE,
    METADATA_PARQUET,
    REPOS_PARQUET,
)

# Reuse fetch infrastructure from 01_fetch_repos.py
from importlib.machinery import SourceFileLoader as _SFL

_fetch_mod = _SFL("_fetch", "01_fetch_repos.py").load_module()
_build_batch_query = _fetch_mod._build_batch_query
_extract_readme = _fetch_mod._extract_readme
_graphql_query = _fetch_mod._graphql_query
README_FRAGMENT = _fetch_mod.README_FRAGMENT

PROGRESS_FILE = DATA_DIR / ".patch_readme_progress.json"
MIN_README_CHARS = 200


def _fetch_readme_batch(repos: list[str]) -> dict[str, str]:
    """Fetch READMEs for a batch using expanded aliases."""
    query = _build_batch_query(repos, README_FRAGMENT)
    try:
        result = _graphql_query(query)
    except RuntimeError:
        if len(repos) <= 5:
            return {}
        half = len(repos) // 2
        print(f"\n  Batch failed, splitting {len(repos)} → {half}+{len(repos)-half}")
        left = _fetch_readme_batch(repos[:half])
        right = _fetch_readme_batch(repos[half:])
        left.update(right)
        return left

    data = result.get("data") or {}
    readmes = {}
    for i in range(len(repos)):
        node = data.get(f"repo{i}")
        if node is None:
            continue
        readmes[node["nameWithOwner"]] = _extract_readme(node)
    return readmes


def _safe_write_parquet(df: pd.DataFrame, path, expected_len: int, expected_cols: set):
    """Write parquet via temp file with verification, then atomic rename."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(DATA_DIR), suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == expected_len, (
            f"Row count mismatch: expected {expected_len}, got {len(verify)}"
        )
        assert set(verify.columns) == expected_cols, (
            f"Column mismatch: {set(verify.columns) ^ expected_cols}"
        )
        os.replace(tmp_path, str(path))
    except Exception:
        os.unlink(tmp_path)
        raise


def main():
    if not METADATA_PARQUET.exists():
        print(f"Error: {METADATA_PARQUET} not found. Run 01_fetch_repos.py first.")
        return

    df_meta = pd.read_parquet(METADATA_PARQUET)
    df_meta["readme"] = df_meta["readme"].fillna("")
    original_cols = set(df_meta.columns)
    original_len = len(df_meta)
    print(f"Loaded {original_len} repos from {METADATA_PARQUET}")

    # Sort by stars to determine the cutoff
    df_meta = df_meta.sort_values("stargazers_count", ascending=False).reset_index(drop=True)

    # Star floor: position at FETCH_OVERSHOOT_COUNT * 1.2 (no point fetching
    # READMEs for repos that can't possibly make the final cut)
    cutoff_idx = min(int(FETCH_OVERSHOOT_COUNT * 1.2), len(df_meta) - 1)
    star_floor = df_meta.loc[cutoff_idx, "stargazers_count"]

    # Find repos with missing/short READMEs above the star floor
    mask = (
        (df_meta["readme"].str.strip().str.len() < MIN_README_CHARS)
        & (df_meta["stargazers_count"] >= star_floor)
    )
    targets = df_meta.loc[mask, "full_name"].tolist()

    # Load checkpoint progress (resume support)
    done = set()
    if PROGRESS_FILE.exists():
        done = set(json.loads(PROGRESS_FILE.read_text()))
        print(f"Resuming: {len(done)} repos already fetched in prior run")

    targets = [t for t in targets if t not in done]
    print(f"Targets: {len(targets)} repos with short/empty READMEs (star floor: {star_floor})")

    if not targets:
        print("Nothing to do — all qualifying repos already have READMEs.")
        return

    # Fetch READMEs in batches
    fetched = {}
    batches = [targets[i : i + GRAPHQL_BATCH_SIZE] for i in range(0, len(targets), GRAPHQL_BATCH_SIZE)]
    print(f"Fetching {len(targets)} repos in {len(batches)} batches...")

    for batch in tqdm(batches, desc="Fetching READMEs"):
        result = _fetch_readme_batch(batch)
        fetched.update(result)
        done.update(batch)
        # Checkpoint after each batch
        PROGRESS_FILE.write_text(json.dumps(sorted(done)))

    # Count recoveries
    recovered = {name: text for name, text in fetched.items() if len(text.strip()) >= MIN_README_CHARS}
    print(f"\nFetched {len(fetched)} repos, recovered {len(recovered)} with substantial READMEs")

    if not recovered:
        print("No new READMEs recovered. Cleaning up.")
        PROGRESS_FILE.unlink(missing_ok=True)
        return

    # Back up metadata.parquet
    backup_meta = DATA_DIR / "metadata_pre_readme_patch.parquet"
    if not backup_meta.exists():
        shutil.copy2(METADATA_PARQUET, backup_meta)
        print(f"Backed up {METADATA_PARQUET} → {backup_meta}")

    # Merge fetched READMEs into metadata
    name_to_idx = dict(zip(df_meta["full_name"], df_meta.index))
    for name, text in fetched.items():
        if name in name_to_idx and text:
            df_meta.loc[name_to_idx[name], "readme"] = text

    _safe_write_parquet(df_meta, METADATA_PARQUET, original_len, original_cols)
    print(f"Updated {METADATA_PARQUET} with {len(recovered)} recovered READMEs")

    # Re-select repos.parquet
    df_sorted = df_meta.sort_values("stargazers_count", ascending=False)
    df_with_readme = df_sorted[df_sorted["readme"].str.strip().str.len() >= MIN_README_CHARS]
    df_final = df_with_readme.head(FETCH_OVERSHOOT_COUNT).reset_index(drop=True)

    # Back up existing repos.parquet
    if REPOS_PARQUET.exists():
        backup_repos = DATA_DIR / "repos_pre_readme_patch.parquet"
        if not backup_repos.exists():
            shutil.copy2(REPOS_PARQUET, backup_repos)
            print(f"Backed up {REPOS_PARQUET} → {backup_repos}")

    _safe_write_parquet(df_final, REPOS_PARQUET, len(df_final), original_cols)

    min_stars = df_final["stargazers_count"].iloc[-1] if len(df_final) > 0 else 0
    print(f"\nRe-selected {len(df_final)} repos → {REPOS_PARQUET} (min stars: {min_stars})")

    # Compare with old repos.parquet
    backup_repos = DATA_DIR / "repos_pre_readme_patch.parquet"
    if backup_repos.exists():
        old_names = set(pd.read_parquet(backup_repos, columns=["full_name"])["full_name"])
        new_names = set(df_final["full_name"])
        added = new_names - old_names
        removed = old_names - new_names
        print(f"  Added: {len(added)} repos, Removed: {len(removed)} repos")
        if added:
            # Show top 10 by stars
            top_added = df_final[df_final["full_name"].isin(added)].nlargest(10, "stargazers_count")
            print("  Top recovered repos:")
            for _, row in top_added.iterrows():
                print(f"    {row['full_name']:40s} {row['stargazers_count']:>7,} ★")

    # Clean up progress file
    PROGRESS_FILE.unlink(missing_ok=True)

    print("\nNext steps — re-run downstream pipeline:")
    print("  python 01b_summarize_readmes.py  # incremental, only new repos")
    print("  python 02_embed_readmes.py       # full re-run")
    print("  python 03_reduce_umap.py         # full re-run")
    print("  python 04_label_topics.py        # full re-run")
    print("  python 05_visualize.py           # full re-run")


if __name__ == "__main__":
    main()
