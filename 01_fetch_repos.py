"""Fetch repo metadata + READMEs via batched GraphQL direct lookups.

Reads candidate repo names from data/candidates.csv (produced by
00_enumerate_repos.py or copied from committed fallback), looks up each
via repository(owner:, name:) queries, and keeps the top TARGET_REPO_COUNT
by stargazer count.
"""

import csv
import json
import shutil
import time

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    CANDIDATES_COMMITTED,
    CANDIDATES_CSV,
    GITHUB_TOKEN,
    GRAPHQL_BATCH_SIZE,
    REPOS_PARQUET,
    TARGET_REPO_COUNT,
)

GRAPHQL_URL = "https://api.github.com/graphql"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json",
}

REPO_FIELDS_FRAGMENT = """
fragment RepoFields on Repository {
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
  readme_md: object(expression: "HEAD:README.md") { ... on Blob { text } }
  readme_lower: object(expression: "HEAD:readme.md") { ... on Blob { text } }
}
"""

CHECKPOINT_EVERY = 50  # batches between checkpoints


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
    for key in ("readme_md", "readme_lower"):
        obj = node.get(key)
        if obj and obj.get("text"):
            return obj["text"]
    return ""


def _parse_node(node: dict) -> dict:
    """Parse a GraphQL Repository node into a flat row dict."""
    topics_list = [
        t["topic"]["name"]
        for t in (node.get("repositoryTopics") or {}).get("nodes", [])
    ]
    default_branch_ref = node.get("defaultBranchRef") or {}
    target = default_branch_ref.get("target") or {}
    history = target.get("history") or {}
    languages_edges = (node.get("languages") or {}).get("edges", [])
    languages_json = json.dumps(
        [{"name": e["node"]["name"], "bytes": e["size"]} for e in languages_edges]
    )

    return {
        "full_name": node["nameWithOwner"],
        "description": node.get("description") or "",
        "language": (node.get("primaryLanguage") or {}).get("name", ""),
        "stargazers_count": node["stargazerCount"],
        "license": (node.get("licenseInfo") or {}).get("spdxId", ""),
        "created_at": node.get("createdAt", ""),
        "topics": ",".join(topics_list),
        "readme": _extract_readme(node),
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
                f"No candidates file found. Run 00_enumerate_repos.py first, "
                f"or place candidates.csv in the repo root."
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


def _build_batch_query(repos: list[str]) -> str:
    """Build a batched GraphQL query with aliased repository() lookups."""
    parts = []
    for i, full_name in enumerate(repos):
        owner, name = full_name.split("/", 1)
        # Escape quotes in owner/name (shouldn't happen, but be safe)
        owner = owner.replace('"', '\\"')
        name = name.replace('"', '\\"')
        parts.append(f'  repo{i}: repository(owner: "{owner}", name: "{name}") {{ ...RepoFields }}')
    query_body = "\n".join(parts)
    return f"query {{\n{query_body}\n}}\n{REPO_FIELDS_FRAGMENT}"


def _fetch_batch(repos: list[str], batch_size: int) -> list[dict]:
    """Fetch a batch, falling back to smaller batches on 502."""
    query = _build_batch_query(repos)
    try:
        result = _graphql_query(query)
    except RuntimeError:
        if batch_size <= 5:
            raise
        # Fall back to smaller batches
        half = len(repos) // 2
        print(f"\n  Batch failed, splitting into two sub-batches of {half}")
        left = _fetch_batch(repos[:half], half)
        right = _fetch_batch(repos[half:], len(repos) - half)
        return left + right

    data = result.get("data") or {}
    rows = []
    for i in range(len(repos)):
        node = data.get(f"repo{i}")
        if node is None:
            # Deleted/renamed/private repo — skip silently
            continue
        try:
            rows.append(_parse_node(node))
        except (KeyError, TypeError) as e:
            print(f"\n  Skipping {repos[i]}: parse error: {e}")
    return rows


def main():
    candidates = _load_candidates()

    # Resume: load existing progress and skip already-fetched repos
    existing_rows = {}
    if REPOS_PARQUET.exists():
        df_existing = pd.read_parquet(REPOS_PARQUET)
        for row in df_existing.to_dict("records"):
            existing_rows[row["full_name"]] = row
        print(f"Resuming: {len(existing_rows)} repos already fetched")

    # Filter candidates to only those not yet fetched
    to_fetch = [c for c in candidates if c not in existing_rows]
    print(f"Need to fetch {len(to_fetch)} remaining candidates")

    if to_fetch:
        new_rows = []
        batches_since_checkpoint = 0

        with tqdm(total=len(to_fetch), desc="Fetching repos") as pbar:
            for batch_start in range(0, len(to_fetch), GRAPHQL_BATCH_SIZE):
                batch = to_fetch[batch_start : batch_start + GRAPHQL_BATCH_SIZE]
                rows = _fetch_batch(batch, len(batch))
                new_rows.extend(rows)
                pbar.update(len(batch))
                batches_since_checkpoint += 1

                # Checkpoint periodically
                if batches_since_checkpoint >= CHECKPOINT_EVERY:
                    _save_checkpoint(existing_rows, new_rows)
                    batches_since_checkpoint = 0

                # Brief pause to stay well within rate limits
                time.sleep(0.5)

        # Merge new rows into existing
        for r in new_rows:
            name = r["full_name"]
            # Carry over extra columns from old data (like 'summary')
            if name in existing_rows:
                for col, val in existing_rows[name].items():
                    if col not in r:
                        r[col] = val
            existing_rows[name] = r

        print(f"Fetched {len(new_rows)} new repos ({len(new_rows) - len([r for r in new_rows])} errors)")

    # Sort by stars descending and keep top TARGET_REPO_COUNT
    all_rows = sorted(existing_rows.values(), key=lambda r: r.get("stargazers_count", 0), reverse=True)
    rows = all_rows[:TARGET_REPO_COUNT]
    if len(all_rows) > TARGET_REPO_COUNT:
        min_stars = rows[-1].get("stargazers_count", 0)
        print(f"Keeping top {TARGET_REPO_COUNT} repos (min stars: {min_stars})")

    df = pd.DataFrame(rows)
    df.to_parquet(REPOS_PARQUET, index=False)
    print(f"Done. Saved {len(df)} repos to {REPOS_PARQUET}")
    print(f"  Non-empty READMEs: {(df['readme'].str.len() > 0).sum()}")


def _save_checkpoint(existing_rows: dict, new_rows: list[dict]):
    """Save intermediate progress to parquet."""
    merged = dict(existing_rows)
    for r in new_rows:
        name = r["full_name"]
        if name in merged:
            for col, val in merged[name].items():
                if col not in r:
                    r[col] = val
        merged[name] = r
    df = pd.DataFrame(list(merged.values()))
    df.to_parquet(REPOS_PARQUET, index=False)
    print(f"\n  Checkpoint: saved {len(df)} repos")


if __name__ == "__main__":
    main()
