"""Fetch lightweight metadata for out-of-set GitHub targets.

Reads data/dependencies.parquet, takes unique target_nwo values where
target_kind == 'external_repo', and fetches description, primaryLanguage,
license, createdAt, pushedAt, stargazerCount, forkCount, isArchived.
Writes data/external_repos.parquet.

Resumable: skips repos already present in the output file.

Cheap relative to stage 08 — no dep-graph lookup, just one batched
GraphQL with aliased repository() reads. Batches of 50 are stable.
"""

import argparse
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from config import DATA_DIR, GITHUB_TOKEN, REPOS_PARQUET
from tqdm import tqdm

DEPS_PARQUET = DATA_DIR / "dependencies.parquet"
EXTERNAL_PARQUET = DATA_DIR / "external_repos.parquet"

BATCH_SIZE = 50
CONCURRENT_WORKERS = 5
CHECKPOINT_EVERY = 20  # batches between disk writes

GRAPHQL_URL = "https://api.github.com/graphql"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json",
}

FRAGMENT = """
fragment ExternalFields on Repository {
  nameWithOwner
  description
  primaryLanguage { name }
  stargazerCount
  forkCount
  licenseInfo { spdxId }
  createdAt
  pushedAt
  isArchived
  isFork
  isMirror
  isDisabled
  owner { __typename }
}
"""


def _graphql(query: str, max_retries: int = 5) -> dict:
    payload = {"query": query}
    last_status = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(GRAPHQL_URL, headers=HEADERS, json=payload, timeout=60)
            last_status = resp.status_code
            remaining = int(resp.headers.get("X-RateLimit-Remaining", 5000))
            if remaining < 200:
                reset_at = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait = max(reset_at - int(time.time()), 10)
                print(f"\n  Rate limit low ({remaining}), waiting {wait}s")
                time.sleep(wait)
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except ValueError:
                    print(f"\n  non-JSON 200 body: {resp.text[:200]}")
                    time.sleep(2 ** (attempt + 1))
                    continue
                if "errors" in body:
                    err = body["errors"][0].get("message", "")
                    if "rate limit" in err.lower() or "timeout" in err.lower():
                        wait = 2 ** (attempt + 2)
                        time.sleep(wait)
                        continue
                return body
            if resp.status_code in (403, 429, 502, 503):
                wait = 2 ** (attempt + 2)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = max(wait, int(retry_after) + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            wait = 2 ** (attempt + 2)
            print(f"\n  {type(e).__name__}: {e}, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"GraphQL failed after {max_retries} retries (last status: {last_status})")


def _build_query(nwos: list[str]) -> str:
    parts = []
    for i, nwo in enumerate(nwos):
        owner, name = nwo.split("/", 1)
        owner = owner.replace('"', '\\"')
        name = name.replace('"', '\\"')
        parts.append(f'  r{i}: repository(owner: "{owner}", name: "{name}") {{ ...ExternalFields }}')
    return f"query {{\n{chr(10).join(parts)}\n}}\n{FRAGMENT}"


def _parse(node: dict) -> dict:
    return {
        "full_name": node["nameWithOwner"],
        "description": node.get("description") or "",
        "primary_language": (node.get("primaryLanguage") or {}).get("name") or "",
        "stargazers_count": int(node.get("stargazerCount") or 0),
        "fork_count": int(node.get("forkCount") or 0),
        "license": (node.get("licenseInfo") or {}).get("spdxId") or "",
        "created_at": node.get("createdAt") or "",
        "pushed_at": node.get("pushedAt") or "",
        "is_archived": bool(node.get("isArchived")),
        "is_fork": bool(node.get("isFork")),
        "is_mirror": bool(node.get("isMirror")),
        "is_disabled": bool(node.get("isDisabled")),
        "owner_type": (node.get("owner") or {}).get("__typename") or "",
    }


def _fetch_batch(nwos: list[str]) -> list[dict]:
    """Fetch a batch, splitting on failure (mirrors stage 01's approach)."""
    query = _build_query(nwos)
    try:
        body = _graphql(query)
    except RuntimeError:
        if len(nwos) <= 5:
            raise
        half = len(nwos) // 2
        print(f"\n  batch of {len(nwos)} failed, splitting")
        return _fetch_batch(nwos[:half]) + _fetch_batch(nwos[half:])

    data = body.get("data") or {}
    rows = []
    for i, nwo in enumerate(nwos):
        node = data.get(f"r{i}")
        if node is None:
            # Repo deleted, renamed, or made private since stage 08 saw it.
            # Record a tombstone so we don't re-attempt.
            rows.append(
                {
                    "full_name": nwo,
                    "description": "",
                    "primary_language": "",
                    "stargazers_count": 0,
                    "fork_count": 0,
                    "license": "",
                    "created_at": "",
                    "pushed_at": "",
                    "is_archived": False,
                    "is_fork": False,
                    "is_mirror": False,
                    "is_disabled": False,
                    "owner_type": "",
                    "fetch_status": "missing",
                }
            )
            continue
        row = _parse(node)
        row["fetch_status"] = "ok"
        rows.append(row)
    return rows


def _safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N repos this run")
    args = parser.parse_args()

    if not GITHUB_TOKEN:
        raise SystemExit("GITHUB_TOKEN not set")
    if not DEPS_PARQUET.exists():
        raise SystemExit(f"{DEPS_PARQUET} not found — run stage 08 first")

    deps = pd.read_parquet(DEPS_PARQUET, columns=["target_nwo", "target_kind"])
    external_targets = deps[deps["target_kind"] == "external_repo"]["target_nwo"].dropna().unique().tolist()
    external_targets.sort()
    print(f"Unique external GitHub targets in stage 08 output: {len(external_targets)}")

    in_set = set()
    if REPOS_PARQUET.exists():
        in_set = set(pd.read_parquet(REPOS_PARQUET, columns=["full_name"])["full_name"])
    # Defensive: any target_kind=external_repo should already not be in_set.
    external_targets = [t for t in external_targets if t not in in_set]

    # Resume
    existing_rows = []
    if EXTERNAL_PARQUET.exists():
        existing_df = pd.read_parquet(EXTERNAL_PARQUET)
        existing_rows = existing_df.to_dict("records")
        already_done = set(existing_df["full_name"])
        print(f"Resuming: {len(already_done)} already enriched")
        external_targets = [t for t in external_targets if t not in already_done]

    if args.limit is not None:
        external_targets = external_targets[: args.limit]
    print(f"To fetch this run: {len(external_targets)} repos")
    if not external_targets:
        print("Nothing to do.")
        return

    batches = [external_targets[i : i + BATCH_SIZE] for i in range(0, len(external_targets), BATCH_SIZE)]

    rows = list(existing_rows)
    lock = threading.Lock()
    batches_since_checkpoint = 0

    def _checkpoint():
        df = pd.DataFrame(rows)
        _safe_write_parquet(df, EXTERNAL_PARQUET)

    with tqdm(total=len(external_targets), desc="Enriching externals") as pbar:
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            futures = {executor.submit(_fetch_batch, b): b for b in batches}
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    batch_rows = future.result()
                except Exception as e:
                    print(f"\n  batch failed permanently: {e}")
                    batch_rows = []
                with lock:
                    rows.extend(batch_rows)
                    pbar.update(len(batch))
                    batches_since_checkpoint += 1
                    if batches_since_checkpoint >= CHECKPOINT_EVERY:
                        _checkpoint()
                        batches_since_checkpoint = 0

    _checkpoint()

    df = pd.DataFrame(rows)
    print(f"\nDone. {len(df)} external repos enriched.")
    if "fetch_status" in df.columns:
        print(df["fetch_status"].value_counts().to_string())
    if "primary_language" in df.columns:
        print("\nTop external-target languages:")
        print(df["primary_language"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
