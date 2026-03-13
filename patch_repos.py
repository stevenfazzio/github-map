"""One-off script to backfill repos.parquet.

Removes repos with short/empty READMEs (<200 chars) and backfills from
metadata.parquet to reach 11K total repos. Safe to delete after use.
"""

import pandas as pd
from config import REPOS_PARQUET, METADATA_PARQUET, TARGET_REPO_COUNT


def main():
    # Load current repos
    repos = pd.read_parquet(REPOS_PARQUET)
    print(f"Loaded {len(repos):,} repos from {REPOS_PARQUET}")

    # Identify short READMEs
    short_mask = repos["readme"].fillna("").str.strip().str.len() < 200
    n_short = short_mask.sum()
    print(f"Found {n_short} repos with README < 200 chars — dropping them")
    repos = repos[~short_mask].copy()
    print(f"After drop: {len(repos):,} repos")

    # Load metadata cache for replacements
    meta = pd.read_parquet(METADATA_PARQUET)
    print(f"Loaded {len(meta):,} repos from {METADATA_PARQUET}")

    # Filter metadata: not already in repos, has sufficient README
    existing_names = set(repos["full_name"])
    meta = meta[~meta["full_name"].isin(existing_names)].copy()
    meta = meta[meta["readme"].fillna("").str.strip().str.len() >= 200].copy()
    print(f"Eligible replacements from metadata: {len(meta):,}")

    # Sort by stars descending, take what we need
    needed = TARGET_REPO_COUNT - len(repos)
    if needed <= 0:
        print(f"Already at {len(repos):,} repos, nothing to backfill")
        return

    if len(meta) < needed:
        print(f"WARNING: Only {len(meta):,} replacements available, need {needed:,}")
        print("Will add all available replacements")

    meta = meta.sort_values("stargazers_count", ascending=False).head(needed)
    print(f"Adding {len(meta):,} repos (min stars: {meta['stargazers_count'].min():,})")

    # Ensure columns match
    missing_cols = set(repos.columns) - set(meta.columns)
    if missing_cols:
        print(f"Note: replacement repos missing columns {missing_cols} — will be NaN")

    # Combine
    patched = pd.concat([repos, meta[repos.columns]], ignore_index=True)

    # Verify
    final_short = patched["readme"].fillna("").str.strip().str.len() < 200
    print(f"\n--- Results ---")
    print(f"Total repos: {len(patched):,}")
    print(f"Repos with README < 200 chars: {final_short.sum()}")
    print(f"Min stars: {patched['stargazers_count'].min():,}")

    # Write patched file first, then replace original
    patched_path = REPOS_PARQUET.with_name("repos_patched.parquet")
    patched.to_parquet(patched_path, index=False)
    print(f"\nWritten to {patched_path}")

    # Replace original with backup
    backup_path = REPOS_PARQUET.with_name("repos_pre_patch.parquet")
    REPOS_PARQUET.rename(backup_path)
    patched_path.rename(REPOS_PARQUET)
    print(f"Backed up original to {backup_path}")
    print(f"Replaced {REPOS_PARQUET} with patched version")


if __name__ == "__main__":
    main()
