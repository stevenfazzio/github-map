"""Fetch top 1K GitHub repos by stars and their READMEs."""

import base64
import time

import pandas as pd
import requests
from tqdm import tqdm

from config import GITHUB_TOKEN, REPOS_PARQUET, TARGET_REPO_COUNT

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
}
SEARCH_URL = "https://api.github.com/search/repositories"
PER_PAGE = 100
# GitHub search caps at 1,000 results per query, so we stay within 9 pages
# and use star-count ranges to paginate beyond that limit.
MAX_PAGES_PER_QUERY = 9


def _parse_item(item: dict) -> dict:
    return {
        "full_name": item["full_name"],
        "description": item.get("description") or "",
        "language": item.get("language") or "",
        "stargazers_count": item["stargazers_count"],
        "license": (item.get("license") or {}).get("spdx_id", ""),
        "created_at": item["created_at"],
        "topics": ",".join(item.get("topics", [])),
    }


def _search_with_retry(params: dict, max_retries: int = 5) -> requests.Response:
    """Make a search request with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        resp = requests.get(SEARCH_URL, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            return resp
        if resp.status_code in (403, 429):
            wait = 2 ** (attempt + 2)  # 4, 8, 16, 32, 64 seconds
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                wait = max(wait, int(retry_after) + 1)
            print(f"\n  Rate limited, waiting {wait}s (attempt {attempt + 1})...")
            time.sleep(wait)
        else:
            resp.raise_for_status()
    resp.raise_for_status()
    return resp  # unreachable, but satisfies type checker


def fetch_repo_list() -> list[dict]:
    """Fetch top repos by star count, using star-range pagination."""
    repos = []
    seen = set()
    star_ceiling = None  # no upper bound for the first query

    with tqdm(total=TARGET_REPO_COUNT, desc="Searching repos") as pbar:
        while len(repos) < TARGET_REPO_COUNT:
            q = f"stars:<={star_ceiling}" if star_ceiling else "stars:>=1"

            for page in range(1, MAX_PAGES_PER_QUERY + 1):
                resp = _search_with_retry(
                    {
                        "q": q,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": PER_PAGE,
                        "page": page,
                    }
                )
                items = resp.json()["items"]
                if not items:
                    break

                for item in items:
                    name = item["full_name"]
                    if name not in seen:
                        seen.add(name)
                        repos.append(_parse_item(item))
                        pbar.update(1)
                        if len(repos) >= TARGET_REPO_COUNT:
                            break

                if len(repos) >= TARGET_REPO_COUNT:
                    break
                time.sleep(3)

            if not items or len(repos) >= TARGET_REPO_COUNT:
                break

            # Next query: stars at or below the last repo's count
            star_ceiling = repos[-1]["stargazers_count"]
            time.sleep(3)

    return repos[:TARGET_REPO_COUNT]


def fetch_readme(full_name: str) -> str:
    """Fetch and decode a repo's README."""
    url = f"https://api.github.com/repos/{full_name}/readme"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return ""
    content = resp.json().get("content", "")
    try:
        return base64.b64decode(content).decode("utf-8", errors="replace")
    except Exception:
        return ""


def main():
    # Resume support: load existing progress
    existing = set()
    if REPOS_PARQUET.exists():
        df_existing = pd.read_parquet(REPOS_PARQUET)
        existing = set(df_existing["full_name"])
        rows = df_existing.to_dict("records")
        print(f"Resuming: {len(rows)} repos already fetched")
    else:
        rows = []

    # Fetch repo list (always refetch to get full list)
    repo_list = fetch_repo_list()
    print(f"Found {len(repo_list)} repos from search")

    # Fetch READMEs for repos we haven't processed yet
    pending = [r for r in repo_list if r["full_name"] not in existing]
    print(f"Fetching READMEs for {len(pending)} new repos")

    for i, repo in enumerate(tqdm(pending, desc="Fetching READMEs")):
        repo["readme"] = fetch_readme(repo["full_name"])
        rows.append(repo)

        # Save progress every 100 repos
        if (i + 1) % 100 == 0:
            pd.DataFrame(rows).to_parquet(REPOS_PARQUET, index=False)
            print(f"  Saved checkpoint ({len(rows)} repos)")

        time.sleep(0.8)  # ~1 req/sec

    df = pd.DataFrame(rows)
    df.to_parquet(REPOS_PARQUET, index=False)
    print(f"Done. Saved {len(df)} repos to {REPOS_PARQUET}")
    print(f"  Non-empty READMEs: {(df['readme'].str.len() > 0).sum()}")


if __name__ == "__main__":
    main()
