"""Fetch repo metadata + READMEs via batched GraphQL direct lookups.

Two-pass approach for speed:
  1. Metadata-only pass: fetch all ~25K candidates with large batches (no README blobs).
  2. README pass: fetch READMEs only for the top FETCH_OVERSHOOT_COUNT repos by stars.

Reads candidate repo names from data/candidates.csv (produced by
00_enumerate_repos.py or copied from committed fallback).
"""

import csv
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    CANDIDATES_COMMITTED,
    CANDIDATES_CSV,
    FETCH_OVERSHOOT_COUNT,
    GITHUB_TOKEN,
    GRAPHQL_BATCH_SIZE,
    METADATA_PARQUET,
    REPOS_PARQUET,
)

CONCURRENT_REQUESTS = 5
METADATA_BATCH_SIZE = 25  # GitHub GraphQL limits query complexity regardless of fields
CHECKPOINT_EVERY = 50  # batches between checkpoints

GRAPHQL_URL = "https://api.github.com/graphql"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json",
}

METADATA_FRAGMENT = """
fragment MetadataFields on Repository {
  nameWithOwner
  description
  primaryLanguage { name }
  stargazerCount
  licenseInfo { spdxId }
  createdAt
  repositoryTopics(first: 20) { nodes { topic { name } } }
  pushedAt
  forkCount
  isArchived
  diskUsage
  hasWikiEnabled
  hasDiscussionsEnabled
  watchers { totalCount }
  issues(states: OPEN) { totalCount }
  pullRequests(states: OPEN) { totalCount }
  releases { totalCount }
  discussions { totalCount }
  fundingLinks { platform url }
  defaultBranchRef {
    name
    target { ... on Commit { history { totalCount } } }
  }
  owner { __typename }
  languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
    edges { size node { name } }
  }
}
"""

README_ALIASES = [
    ("readme_md", "README.md"),
    ("readme_lower", "readme.md"),
    ("readme_rst", "README.rst"),
    ("readme_txt", "README.txt"),
    ("readme_bare", "README"),
    ("readme_title", "Readme.md"),
    ("readme_markdown", "README.markdown"),
    ("readme_rst_lower", "readme.rst"),
    ("readme_txt_lower", "readme.txt"),
    ("readme_bare_lower", "readme"),
    ("readme_titlecase", "ReadMe.md"),
]


def _build_readme_fragment() -> str:
    """Build the ReadmeFields fragment from README_ALIASES."""
    lines = ["fragment ReadmeFields on Repository {", "  nameWithOwner"]
    for alias, filename in README_ALIASES:
        lines.append(f'  {alias}: object(expression: "HEAD:{filename}") {{ ... on Blob {{ text }} }}')
    lines.append("}")
    return "\n".join(lines)


README_FRAGMENT = _build_readme_fragment()


def _graphql_query(query: str, variables: dict | None = None, max_retries: int = 5) -> dict:
    """Execute a GraphQL query with rate-limit-aware retry."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json=payload,
                timeout=60,
            )
            # Check rate limit budget proactively
            remaining = int(resp.headers.get("X-RateLimit-Remaining", 5000))
            if remaining < 500:
                reset_at = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait = max(reset_at - int(time.time()), 10)
                print(f"\n  Rate limit low ({remaining} remaining), waiting {wait}s...")
                time.sleep(wait)

            if resp.status_code == 200:
                body = resp.json()
                if "errors" in body:
                    err_msg = body["errors"][0].get("message", "")
                    if "rate limit" in err_msg.lower() or "timeout" in err_msg.lower():
                        wait = 2 ** (attempt + 2)
                        print(f"\n  GraphQL error: {err_msg}, retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    print(f"\n  GraphQL error: {err_msg}")
                return body
            if resp.status_code in (403, 429, 502, 503):
                wait = 2 ** (attempt + 2)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = max(wait, int(retry_after) + 1)
                print(f"\n  HTTP {resp.status_code}, waiting {wait}s (attempt {attempt + 1})...")
                time.sleep(wait)
            else:
                resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            wait = 2 ** (attempt + 2)
            print(f"\n  {type(e).__name__}, retrying in {wait}s (attempt {attempt + 1})...")
            time.sleep(wait)
    raise RuntimeError(f"GraphQL query failed after {max_retries} retries")


def _extract_readme(node: dict) -> str:
    """Extract README text from GraphQL aliases, taking the first non-null."""
    for alias, _ in README_ALIASES:
        obj = node.get(alias)
        if obj and obj.get("text"):
            return obj["text"]
    return ""


def _parse_metadata(node: dict) -> dict:
    """Parse a GraphQL Repository node into a flat row dict (no README)."""
    topics_list = [t["topic"]["name"] for t in (node.get("repositoryTopics") or {}).get("nodes", [])]
    default_branch_ref = node.get("defaultBranchRef") or {}
    target = default_branch_ref.get("target") or {}
    history = target.get("history") or {}
    languages_edges = (node.get("languages") or {}).get("edges", [])
    languages_json = json.dumps([{"name": e["node"]["name"], "bytes": e["size"]} for e in languages_edges])

    return {
        "full_name": node["nameWithOwner"],
        "description": node.get("description") or "",
        "language": (node.get("primaryLanguage") or {}).get("name", ""),
        "stargazers_count": node["stargazerCount"],
        "license": (node.get("licenseInfo") or {}).get("spdxId", ""),
        "created_at": node.get("createdAt", ""),
        "topics": ",".join(topics_list),
        "pushed_at": node.get("pushedAt") or "",
        "fork_count": node.get("forkCount", 0),
        "is_archived": node.get("isArchived", False),
        "disk_usage_kb": node.get("diskUsage", 0),
        "has_wiki": node.get("hasWikiEnabled", False),
        "has_discussions": node.get("hasDiscussionsEnabled", False),
        "watcher_count": (node.get("watchers") or {}).get("totalCount", 0),
        "open_issue_count": (node.get("issues") or {}).get("totalCount", 0),
        "open_pr_count": (node.get("pullRequests") or {}).get("totalCount", 0),
        "release_count": (node.get("releases") or {}).get("totalCount", 0),
        "discussion_count": (node.get("discussions") or {}).get("totalCount", 0),
        "has_funding": len(node.get("fundingLinks") or []) > 0,
        "default_branch": default_branch_ref.get("name", ""),
        "commit_count": history.get("totalCount", 0),
        "owner_type": (node.get("owner") or {}).get("__typename", ""),
        "languages_json": languages_json,
    }


def _load_candidates() -> list[str]:
    """Load candidate repo names from CSV, falling back to committed file."""
    if not CANDIDATES_CSV.exists():
        if CANDIDATES_COMMITTED.exists():
            print(f"Copying committed {CANDIDATES_COMMITTED} → {CANDIDATES_CSV}")
            CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CANDIDATES_COMMITTED, CANDIDATES_CSV)
        else:
            raise FileNotFoundError(
                "No candidates file found. Run 00_enumerate_repos.py first, or place candidates.csv in the repo root."
            )

    candidates = []
    with open(CANDIDATES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["full_name"].strip()
            if "/" in name:
                candidates.append(name)
    print(f"Loaded {len(candidates)} candidates from {CANDIDATES_CSV}")
    return candidates


def _build_batch_query(repos: list[str], fragment: str) -> str:
    """Build a batched GraphQL query with aliased repository() lookups."""
    # Determine fragment name from content
    if "MetadataFields" in fragment:
        spread = "MetadataFields"
    else:
        spread = "ReadmeFields"

    parts = []
    for i, full_name in enumerate(repos):
        owner, name = full_name.split("/", 1)
        owner = owner.replace('"', '\\"')
        name = name.replace('"', '\\"')
        parts.append(f'  repo{i}: repository(owner: "{owner}", name: "{name}") {{ ...{spread} }}')
    query_body = "\n".join(parts)
    return f"query {{\n{query_body}\n}}\n{fragment}"


def _fetch_metadata_batch(repos: list[str]) -> list[dict]:
    """Fetch metadata for a batch (no READMEs), splitting on failure."""
    query = _build_batch_query(repos, METADATA_FRAGMENT)
    try:
        result = _graphql_query(query)
    except RuntimeError:
        if len(repos) <= 5:
            raise
        half = len(repos) // 2
        print(f"\n  Metadata batch failed, splitting into two sub-batches of {half}")
        left = _fetch_metadata_batch(repos[:half])
        right = _fetch_metadata_batch(repos[half:])
        return left + right

    data = result.get("data") or {}
    rows = []
    for i in range(len(repos)):
        node = data.get(f"repo{i}")
        if node is None:
            continue
        try:
            rows.append(_parse_metadata(node))
        except (KeyError, TypeError) as e:
            print(f"\n  Skipping {repos[i]}: parse error: {e}")
    return rows


def _fetch_readme_batch(repos: list[str]) -> dict[str, str]:
    """Fetch READMEs for a batch, returning {full_name: readme_text}."""
    query = _build_batch_query(repos, README_FRAGMENT)
    try:
        result = _graphql_query(query)
    except RuntimeError:
        if len(repos) <= 5:
            raise
        half = len(repos) // 2
        print(f"\n  README batch failed, splitting into two sub-batches of {half}")
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


def _fetch_concurrent(items, batch_size, fetch_fn, desc, combine_fn):
    """Generic concurrent batch fetcher with checkpointing."""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])

    results = []
    lock = threading.Lock()

    with tqdm(total=len(items), desc=desc) as pbar:
        with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
            futures = {executor.submit(fetch_fn, batch): batch for batch in batches}
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"\n  Batch failed permanently: {e}")
                    result = combine_fn()  # empty result
                with lock:
                    results.append(result)
                    pbar.update(len(batch))

    return results


def main():
    candidates = _load_candidates()

    # Resume: load existing metadata (full candidate pool, not just top N)
    existing_rows = {}
    if METADATA_PARQUET.exists():
        df_existing = pd.read_parquet(METADATA_PARQUET)
        for row in df_existing.to_dict("records"):
            existing_rows[row["full_name"]] = row
        print(f"Resuming: {len(existing_rows)} repos in metadata cache")
    elif REPOS_PARQUET.exists():
        # Migrate from old layout where repos.parquet held everything
        df_existing = pd.read_parquet(REPOS_PARQUET)
        for row in df_existing.to_dict("records"):
            existing_rows[row["full_name"]] = row
        print(f"Migrated {len(existing_rows)} repos from {REPOS_PARQUET}")

    # --- Pass 1: Metadata (no READMEs) for all candidates ---
    need_metadata = [c for c in candidates if c not in existing_rows]
    print(f"Pass 1 — metadata: {len(need_metadata)} candidates to fetch")

    if need_metadata:
        batch_results = _fetch_concurrent(
            need_metadata,
            METADATA_BATCH_SIZE,
            _fetch_metadata_batch,
            "Fetching metadata",
            list,
        )
        for batch_rows in batch_results:
            for r in batch_rows:
                name = r["full_name"]
                if name in existing_rows:
                    for col, val in existing_rows[name].items():
                        if col not in r:
                            r[col] = val
                existing_rows[name] = r

    # Save full metadata cache (preserves all candidates for re-selection)
    df_meta = pd.DataFrame(list(existing_rows.values()))
    df_meta.to_parquet(METADATA_PARQUET, index=False)
    print(f"Pass 1 complete: {len(existing_rows)} total repos in metadata cache")

    # --- Sort all repos by stars ---
    all_rows = sorted(
        existing_rows.values(),
        key=lambda r: r.get("stargazers_count", 0),
        reverse=True,
    )

    # --- Pass 2: Fetch READMEs for an expanded pool ---
    # We need FETCH_OVERSHOOT_COUNT repos with non-empty READMEs. Overshoot by
    # 20% to account for repos without READMEs, then expand further if needed.
    pool_size = min(int(FETCH_OVERSHOOT_COUNT * 1.2), len(all_rows))
    pool = all_rows[:pool_size]

    need_readme = [r["full_name"] for r in pool if not r.get("readme")]
    print(f"Pass 2 — READMEs: {len(need_readme)} repos need READMEs (pool: {pool_size})")

    if need_readme:
        batch_results = _fetch_concurrent(
            need_readme,
            GRAPHQL_BATCH_SIZE,
            _fetch_readme_batch,
            "Fetching READMEs",
            dict,
        )
        for readme_map in batch_results:
            for name, text in readme_map.items():
                if name in existing_rows:
                    existing_rows[name]["readme"] = text

        # Update metadata cache with READMEs
        df_meta = pd.DataFrame(list(existing_rows.values()))
        df_meta.to_parquet(METADATA_PARQUET, index=False)

    # --- Final selection: top N repos with usable READMEs ---
    # Filter out empty and very short READMEs (< 200 chars) so downstream
    # scripts don't need to drop repos and reduce the count below target.
    MIN_README_CHARS = 200
    final_rows = []
    for r in all_rows:
        r.setdefault("readme", "")
        if len(r["readme"].strip()) >= MIN_README_CHARS:
            final_rows.append(r)
            if len(final_rows) >= FETCH_OVERSHOOT_COUNT:
                break

    if len(final_rows) < FETCH_OVERSHOOT_COUNT:
        print(f"Warning: only {len(final_rows)} repos have non-empty READMEs")

    min_stars = final_rows[-1].get("stargazers_count", 0) if final_rows else 0
    print(f"Selected {len(final_rows)} repos with READMEs (min stars: {min_stars})")

    df = pd.DataFrame(final_rows)
    df.to_parquet(REPOS_PARQUET, index=False)
    print(f"Done. Saved {len(df)} repos to {REPOS_PARQUET}")


if __name__ == "__main__":
    main()
