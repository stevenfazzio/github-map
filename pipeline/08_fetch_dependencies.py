"""Fetch dependency-graph edges for top 10K repos via GitHub GraphQL.

Writes two outputs:
  data/dependencies.parquet     — one row per dep occurrence
  data/dependency_sources.parquet — one row per source repo with fetch status

Resumable: source repos already present in dependency_sources.parquet are
skipped on subsequent runs. Repos that returned zero manifests, hit a
GraphQL `timedout`, or exhausted retries are recorded with a status so
they don't get re-fetched indefinitely.

Tuning notes from the probe (experiments/probe_dependencies.py):
  - first: 3 manifests x first: 25 deps (~75 nodes) is the largest reliable
    query. first: 10 x first: 50 (500 nodes) triggers 502/timedout on
    monorepos like vscode (220 manifests) and tensorflow (37 manifests).
  - Some repos have totalCount: 0 even though they obviously have a
    package.json. GitHub doesn't index everything; this is a real coverage
    gap, not a bug.
  - For manifests with >25 deps we capture only the first page and record
    manifest_total_deps; a follow-up pass can fill in the long tail.

DATA LIMITATIONS — coverage states per source:
  Per-source completeness is reflected by (status, was_truncated,
  manifests_total, manifest_pages_fetched). Three cases of incomplete
  coverage exist:

  1. Cap fired (was_truncated=True, status="ok"). MAX_MANIFEST_PAGES=10
     caps each source at 30 manifests max (10 pages * MANIFEST_FIRST).
     Without this cap, mega-monorepos like microsoft/vscode (220
     manifests, mostly .github/workflows/*.yml) would block one worker
     for ~6 minutes via sequential pagination. The trailing manifests
     for monorepos are almost entirely additional workflow files
     producing duplicate edges to the same actions (actions/checkout,
     etc.), so the cap loses edge multiplicity, not edge existence.

  2. GitHub timeout mid-pagination (status="timedout",
     manifest_pages_fetched > 0). GitHub's dep-graph index served the
     first N pages then timed out. We have edges from those N pages
     and stop. Affects mega-monorepos like Azure/azure-sdk-for-python
     (1395 manifests, 12 pages captured before timeout).

  3. Index unavailable (status="timedout", manifest_pages_fetched == 0)
     or (status="ok", manifests_total == 0). GitHub didn't index the
     dep manifests for this repo at all. Real coverage gap, not
     fixable from our side.

  Identify affected sources via:
    complete:  status == "ok" AND was_truncated == False
    capped:    was_truncated == True
    partial:   status == "timedout" AND manifest_pages_fetched > 0
    empty:     manifests_total == 0  (or pages_fetched == 0)

  Existing rows fetched before MAX_MANIFEST_PAGES was set to 10 have
  was_truncated=False and were collected with the effectively-uncapped
  MAX_MANIFEST_PAGES=200 — those are complete unless status indicates
  otherwise.
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

MANIFEST_FIRST = 3
DEP_FIRST = 25
CONCURRENT_WORKERS = 5
CHECKPOINT_EVERY = 25  # source repos between checkpoint writes
MAX_MANIFEST_PAGES = 10  # cap: 30 manifests per repo. See module docstring.

DEPS_PARQUET = DATA_DIR / "dependencies.parquet"
SOURCES_PARQUET = DATA_DIR / "dependency_sources.parquet"

GRAPHQL_URL = "https://api.github.com/graphql"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json",
}

QUERY = """
query DepGraph($owner: String!, $name: String!, $manAfter: String) {
  rateLimit { remaining resetAt }
  repository(owner: $owner, name: $name) {
    nameWithOwner
    dependencyGraphManifests(first: %d, after: $manAfter) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        filename
        blobPath
        parseable
        exceedsMaxSize
        dependenciesCount
        dependencies(first: %d) {
          totalCount
          pageInfo { hasNextPage }
          nodes {
            packageName
            packageManager
            requirements
            hasDependencies
            repository {
              nameWithOwner
              stargazerCount
            }
          }
        }
      }
    }
  }
}
""" % (MANIFEST_FIRST, DEP_FIRST)


def _graphql(query: str, variables: dict, max_retries: int = 6) -> dict:
    """Execute a GraphQL query with retry. Returns body dict (may contain
    `errors`); raises only after exhausting retries."""
    payload = {"query": query, "variables": variables}
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
                err_msg = ""
                if "errors" in body:
                    err_msg = body["errors"][0].get("message", "")
                err_lower = err_msg.lower()
                # `timedout` on the dep-graph field is a real coverage limit —
                # GitHub's index can't serve this repo. Retrying yields the
                # same error. Return immediately so the caller can record it.
                if "timedout" in err_lower:
                    return body
                if "rate limit" in err_lower:
                    wait = 2 ** (attempt + 2)
                    print(f"\n  rate-limit GraphQL error, waiting {wait}s")
                    time.sleep(wait)
                    continue
                # Generic transient GraphQL errors ("Something went wrong...")
                # — retry on the next iteration with backoff.
                if err_msg and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 2)
                    print(f"\n  transient GraphQL error: {err_msg[:80]}, waiting {wait}s")
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


def _classify_target(target_nwo: str | None, in_set: set[str]) -> str:
    if target_nwo is None:
        return "unresolved"
    if target_nwo in in_set:
        return "in_set"
    return "external_repo"


def _parse_manifests(repo_node: dict, source_nwo: str, in_set: set[str]) -> list[dict]:
    """Extract one edge row per dependency occurrence."""
    rows = []
    manifests = (repo_node.get("dependencyGraphManifests") or {}).get("nodes") or []
    for m in manifests:
        dep_nodes = (m.get("dependencies") or {}).get("nodes") or []
        for d in dep_nodes:
            target = d.get("repository") or {}
            target_nwo = target.get("nameWithOwner")
            target_stars = target.get("stargazerCount")
            rows.append(
                {
                    "source_nwo": source_nwo,
                    "manifest_filename": m.get("filename") or "",
                    "manifest_path": m.get("blobPath") or "",
                    "manifest_parseable": bool(m.get("parseable")),
                    "manifest_total_deps": int(m.get("dependenciesCount") or 0),
                    "package_name": d.get("packageName") or "",
                    "package_manager": d.get("packageManager") or "",
                    "requirements": d.get("requirements") or "",
                    "has_dependencies": bool(d.get("hasDependencies")),
                    "target_nwo": target_nwo,
                    "target_stars": int(target_stars) if target_stars is not None else None,
                    "target_kind": _classify_target(target_nwo, in_set),
                }
            )
    return rows


def fetch_one_repo(full_name: str, in_set: set[str]) -> tuple[list[dict], dict]:
    """Fetch all dependency edges for one source repo, paginating manifests.

    Returns (edge_rows, source_status_row).
    """
    owner, name = full_name.split("/", 1)
    edges: list[dict] = []
    manifests_total = 0
    pages_fetched = 0
    error_msg: str | None = None
    cursor: str | None = None
    was_truncated = False

    for _ in range(MAX_MANIFEST_PAGES):
        try:
            body = _graphql(QUERY, {"owner": owner, "name": name, "manAfter": cursor})
        except RuntimeError as e:
            error_msg = f"retries_exhausted: {e}"
            break

        if "errors" in body:
            err = body["errors"][0].get("message", "")
            err_lower = err.lower()
            if "timedout" in err_lower:
                error_msg = "timedout"
            elif "rate limit" in err_lower:
                error_msg = "rate_limit"
            else:
                error_msg = "github_error"
            break

        repo_node = (body.get("data") or {}).get("repository")
        if repo_node is None:
            error_msg = error_msg or "null_repository"
            break

        manifest_conn = repo_node.get("dependencyGraphManifests") or {}
        manifests_total = manifest_conn.get("totalCount", 0)
        pages_fetched += 1
        edges.extend(_parse_manifests(repo_node, full_name, in_set))

        page_info = manifest_conn.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if cursor is None:
            break
    else:
        # Loop exhausted without breaking — we hit the page cap with more
        # pages still available. Status is still "ok" (the data we got is
        # valid), but flag the truncation so consumers can detect partial
        # coverage.
        was_truncated = True

    status_row = {
        "source_nwo": full_name,
        "manifests_total": manifests_total,
        "manifest_pages_fetched": pages_fetched,
        "edges_recorded": len(edges),
        "was_truncated": was_truncated,
        "status": error_msg or "ok",
    }
    return edges, status_row


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


def _load_completed() -> tuple[list[dict], list[dict]]:
    """Load existing edges and source status. Returns (edges, source_rows)."""
    edges = []
    sources = []
    if DEPS_PARQUET.exists():
        edges = pd.read_parquet(DEPS_PARQUET).to_dict("records")
    if SOURCES_PARQUET.exists():
        sources = pd.read_parquet(SOURCES_PARQUET).to_dict("records")
    return edges, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N source repos this run (smoke testing)",
    )
    args = parser.parse_args()

    if not GITHUB_TOKEN:
        raise SystemExit("GITHUB_TOKEN not set")
    if not REPOS_PARQUET.exists():
        raise SystemExit(f"{REPOS_PARQUET} not found — run earlier pipeline stages first")

    repos_df = pd.read_parquet(REPOS_PARQUET, columns=["full_name"])
    in_set = set(repos_df["full_name"])
    all_sources = sorted(in_set)  # deterministic for resume
    print(f"Loaded {len(in_set)} source repos")

    edges, sources = _load_completed()
    completed = {s["source_nwo"] for s in sources}
    print(f"Resuming: {len(completed)} already-fetched, {len(edges)} edges in cache")

    todo = [r for r in all_sources if r not in completed]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"To fetch this run: {len(todo)} repos")
    if not todo:
        print("Nothing to do.")
        return

    lock = threading.Lock()
    new_since_checkpoint = 0

    def _checkpoint():
        df_e = pd.DataFrame(edges)
        df_s = pd.DataFrame(sources)
        if not df_e.empty:
            _safe_write_parquet(df_e, DEPS_PARQUET)
        _safe_write_parquet(df_s, SOURCES_PARQUET)

    with tqdm(total=len(todo), desc="Fetching deps") as pbar:
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            futures = {executor.submit(fetch_one_repo, r, in_set): r for r in todo}
            for future in as_completed(futures):
                source = futures[future]
                try:
                    new_edges, status_row = future.result()
                except Exception as e:
                    print(f"\n  unexpected error on {source}: {e}")
                    new_edges = []
                    status_row = {
                        "source_nwo": source,
                        "manifests_total": 0,
                        "manifest_pages_fetched": 0,
                        "edges_recorded": 0,
                        "was_truncated": False,
                        "status": f"unexpected: {type(e).__name__}: {e}",
                    }
                with lock:
                    edges.extend(new_edges)
                    sources.append(status_row)
                    new_since_checkpoint += 1
                    pbar.update(1)
                    if new_since_checkpoint >= CHECKPOINT_EVERY:
                        _checkpoint()
                        new_since_checkpoint = 0

    _checkpoint()

    # Brief summary
    df_s = pd.DataFrame(sources)
    df_e = pd.DataFrame(edges) if edges else pd.DataFrame()
    print(f"\nDone. {len(df_s)} sources processed, {len(df_e)} edges captured.")
    print("\nSource status breakdown:")
    print(df_s["status"].value_counts().to_string())
    if not df_e.empty:
        print("\nTarget kind breakdown:")
        print(df_e["target_kind"].value_counts().to_string())
        print("\nTop ecosystems:")
        print(df_e["package_manager"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
