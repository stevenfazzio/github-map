"""Fetch top 1K GitHub repos by stars via GraphQL (metadata + README in one query)."""

import json
import time

import pandas as pd
import requests
from tqdm import tqdm

from config import GITHUB_TOKEN, REPOS_PARQUET, TARGET_REPO_COUNT

GRAPHQL_URL = "https://api.github.com/graphql"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json",
}
PER_PAGE = 100

QUERY = """
query ($queryString: String!, $cursor: String) {
  search(query: $queryString, type: REPOSITORY, first: 100, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        ... on Repository {
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
          readme_rst: object(expression: "HEAD:README.rst") { ... on Blob { text } }
          readme_txt: object(expression: "HEAD:README.txt") { ... on Blob { text } }
          readme_noext: object(expression: "HEAD:README") { ... on Blob { text } }
        }
      }
    }
  }
}
"""


def _graphql_query(query: str, variables: dict, max_retries: int = 5) -> dict:
    """Execute a GraphQL query with rate-limit-aware retry."""
    for attempt in range(max_retries):
        resp = requests.post(
            GRAPHQL_URL,
            headers=HEADERS,
            json={"query": query, "variables": variables},
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
                # Some GraphQL errors are transient
                err_msg = body["errors"][0].get("message", "")
                if "rate limit" in err_msg.lower() or "timeout" in err_msg.lower():
                    wait = 2 ** (attempt + 2)
                    print(f"\n  GraphQL error: {err_msg}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                # Non-transient errors: print but return what we have
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
    resp.raise_for_status()
    return {}  # unreachable


def _extract_readme(node: dict) -> str:
    """Extract README text from GraphQL aliases, taking the first non-null."""
    for key in ("readme_md", "readme_lower", "readme_rst", "readme_txt", "readme_noext"):
        obj = node.get(key)
        if obj and obj.get("text"):
            return obj["text"]
    return ""


def _parse_node(node: dict) -> dict:
    """Parse a GraphQL Repository node into a flat row dict."""
    # Core fields (backward-compatible with REST version)
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
        # Existing columns
        "full_name": node["nameWithOwner"],
        "description": node.get("description") or "",
        "language": (node.get("primaryLanguage") or {}).get("name", ""),
        "stargazers_count": node["stargazerCount"],
        "license": (node.get("licenseInfo") or {}).get("spdxId", ""),
        "created_at": node.get("createdAt", ""),
        "topics": ",".join(topics_list),
        "readme": _extract_readme(node),
        # New columns
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


def fetch_repo_list() -> list[dict]:
    """Fetch top repos by star count using GraphQL with star-range pagination."""
    repos = []
    seen = set()
    star_ceiling = None

    with tqdm(total=TARGET_REPO_COUNT, desc="Fetching repos") as pbar:
        while len(repos) < TARGET_REPO_COUNT:
            q = f"stars:<={star_ceiling}" if star_ceiling else "stars:>=1"
            q += " sort:stars-desc"
            cursor = None

            while len(repos) < TARGET_REPO_COUNT:
                variables = {"queryString": q, "cursor": cursor}
                result = _graphql_query(QUERY, variables)

                search = result.get("data", {}).get("search", {})
                edges = search.get("edges", [])
                if not edges:
                    break

                for edge in edges:
                    node = edge.get("node")
                    if not node or "nameWithOwner" not in node:
                        continue
                    name = node["nameWithOwner"]
                    if name not in seen:
                        seen.add(name)
                        repos.append(_parse_node(node))
                        pbar.update(1)
                        if len(repos) >= TARGET_REPO_COUNT:
                            break

                page_info = search.get("pageInfo", {})
                if not page_info.get("hasNextPage") or len(repos) >= TARGET_REPO_COUNT:
                    break
                cursor = page_info["endCursor"]
                time.sleep(1)

            if not edges or len(repos) >= TARGET_REPO_COUNT:
                break

            # Next star range: at or below the last repo's count
            star_ceiling = repos[-1]["stargazers_count"]
            time.sleep(1)

    return repos[:TARGET_REPO_COUNT]


# Default values for new columns (used when backfilling old parquet files)
_NEW_COLUMN_DEFAULTS = {
    "pushed_at": "",
    "fork_count": 0,
    "is_archived": False,
    "disk_usage_kb": 0,
    "has_wiki": False,
    "has_discussions": False,
    "watcher_count": 0,
    "open_issue_count": 0,
    "open_pr_count": 0,
    "release_count": 0,
    "discussion_count": 0,
    "has_funding": False,
    "default_branch": "",
    "commit_count": 0,
    "owner_type": "",
    "languages_json": "[]",
}


def main():
    # Resume support: load existing progress
    existing = set()
    if REPOS_PARQUET.exists():
        df_existing = pd.read_parquet(REPOS_PARQUET)
        # Backfill missing new columns with defaults
        for col, default in _NEW_COLUMN_DEFAULTS.items():
            if col not in df_existing.columns:
                df_existing[col] = default
        existing = set(df_existing["full_name"])
        rows = df_existing.to_dict("records")
        print(f"Resuming: {len(rows)} repos already fetched")
    else:
        rows = []

    # Fetch repo list (always refetch to get full list)
    repo_list = fetch_repo_list()
    print(f"Found {len(repo_list)} repos from search")

    # Add repos we haven't seen yet
    new_count = 0
    for repo in repo_list:
        if repo["full_name"] not in existing:
            rows.append(repo)
            new_count += 1

    if new_count:
        print(f"Added {new_count} new repos")

    df = pd.DataFrame(rows)
    df.to_parquet(REPOS_PARQUET, index=False)
    print(f"Done. Saved {len(df)} repos to {REPOS_PARQUET}")
    print(f"  Non-empty READMEs: {(df['readme'].str.len() > 0).sum()}")


if __name__ == "__main__":
    main()
