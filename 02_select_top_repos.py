"""Trim repos.parquet to top N repositories by star count."""

import shutil

import pandas as pd

from config import REPOS_PARQUET, REPOS_PRETRIM_PARQUET, TARGET_REPO_COUNT


def main():
    df = pd.read_parquet(REPOS_PARQUET)
    print(f"Loaded {len(df)} repos")

    if len(df) <= TARGET_REPO_COUNT:
        print(f"Already at or below target ({TARGET_REPO_COUNT:,}), nothing to trim")
        return

    df = df.sort_values("stargazers_count", ascending=False).head(TARGET_REPO_COUNT).reset_index(drop=True)
    shutil.copy2(REPOS_PARQUET, REPOS_PRETRIM_PARQUET)
    print(f"Backed up original ({len(pd.read_parquet(REPOS_PRETRIM_PARQUET)):,} rows) to {REPOS_PRETRIM_PARQUET}")
    df.to_parquet(REPOS_PARQUET, index=False)
    print(
        f"Trimmed to top {TARGET_REPO_COUNT:,} by stars"
        f" (min: {df['stargazers_count'].min():,}), saved back to {REPOS_PARQUET}"
    )


if __name__ == "__main__":
    main()
