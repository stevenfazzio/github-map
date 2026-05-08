"""Probe: gauge feasibility and shape of GitHub dependency graph data.

Picks one top-starred repo per major language, queries GitHub's
dependencyGraphManifests, and reports metrics needed to size a full crawl:
  - Manifest counts, parseable rate (does GitHub know how to read the repo?)
  - Total deps per repo and ecosystem distribution
  - Resolution rate (deps with non-null repository field)
  - In-set rate (resolved targets also in our top-10K)
  - GraphQL rate-limit cost per query (extrapolates to 10K-repo cost)

Saves raw responses to data/experiments/dependencies_probe.json for inspection.

Run: python experiments/probe_dependencies.py
"""

import json
import sys
import time

import pandas as pd
import requests

from pipeline.config import EXPERIMENTS_DIR, GITHUB_TOKEN, REPOS_PARQUET

GRAPHQL_URL = "https://api.github.com/graphql"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json",
}

# One repo from each — picked to span major ecosystems and to surface
# expected coverage gaps (Shell/HTML often have no manifests at all).
SAMPLE_LANGUAGES = [
    "TypeScript",
    "JavaScript",
    "Python",
    "Go",
    "Rust",
    "Java",
    "C++",
    "Ruby",
    "Shell",
    "HTML",
]

# The top-starred repo per language is often an awesome-list with no manifests.
# These extras are picked to make sure we exercise the dep-rich case too. Any
# that aren't in the top 10K are silently skipped.
EXTRA_PROBES = [
    "microsoft/vscode",
    "kubernetes/kubernetes",
    "pallets/flask",
    "rust-lang/cargo",
]

# Empirically: 500-node queries (first:10 x first:50) trigger 502/timedout on
# monorepos. 75 nodes (first:3 x first:25) works on every repo we tested,
# including vscode (220 manifests) and k8s (39). totalCount shows what we
# missed; the production fetch will paginate.
MANIFEST_FIRST = 3
DEP_FIRST = 25

DEPENDENCY_QUERY = f"""
query DepGraph($owner: String!, $name: String!) {{
  rateLimit {{ cost remaining resetAt }}
  repository(owner: $owner, name: $name) {{
    nameWithOwner
    dependencyGraphManifests(first: {MANIFEST_FIRST}) {{
      totalCount
      nodes {{
        filename
        blobPath
        parseable
        exceedsMaxSize
        dependenciesCount
        dependencies(first: {DEP_FIRST}) {{
          totalCount
          nodes {{
            packageName
            packageManager
            requirements
            hasDependencies
            repository {{
              nameWithOwner
              stargazerCount
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""

PROBE_OUTPUT = EXPERIMENTS_DIR / "dependencies_probe.json"


def graphql(query: str, variables: dict, max_retries: int = 4) -> dict:
    payload = {"query": query, "variables": variables}
    last_status = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(GRAPHQL_URL, headers=HEADERS, json=payload, timeout=60)
            last_status = resp.status_code
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    print(f"  non-JSON 200 body: {resp.text[:200]}")
            elif resp.status_code in (403, 429, 502, 503):
                wait = 2 ** (attempt + 2)
                print(f"  HTTP {resp.status_code}, waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            wait = 2 ** (attempt + 2)
            print(f"  {type(e).__name__}: {e}, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"GraphQL probe failed after retries (last status: {last_status})")


def pick_probe_repos(repos_df: pd.DataFrame) -> list[str]:
    """Top-starred per language plus a few hand-picked code-bearing extras."""
    chosen, seen = [], set()
    for lang in SAMPLE_LANGUAGES:
        subset = repos_df[repos_df["language"] == lang]
        if subset.empty:
            continue
        nwo = subset.sort_values("stargazers_count", ascending=False).iloc[0]["full_name"]
        if nwo not in seen:
            chosen.append(nwo)
            seen.add(nwo)
    in_set = set(repos_df["full_name"])
    for nwo in EXTRA_PROBES:
        if nwo in in_set and nwo not in seen:
            chosen.append(nwo)
            seen.add(nwo)
    return chosen


def summarize(repo_data: dict, in_set: set[str]) -> dict:
    manifest_conn = repo_data.get("dependencyGraphManifests") or {}
    manifests = manifest_conn.get("nodes") or []

    deps_total = deps_resolved = deps_in_set = unparseable = 0
    ecosystems: dict[str, int] = {}
    for m in manifests:
        if not m.get("parseable"):
            unparseable += 1
        for d in (m.get("dependencies") or {}).get("nodes") or []:
            deps_total += 1
            pm = d.get("packageManager") or "UNKNOWN"
            ecosystems[pm] = ecosystems.get(pm, 0) + 1
            target = d.get("repository") or {}
            target_nwo = target.get("nameWithOwner")
            if target_nwo:
                deps_resolved += 1
                if target_nwo in in_set:
                    deps_in_set += 1

    return {
        "manifests_total": manifest_conn.get("totalCount", 0),
        "manifests_fetched": len(manifests),
        "manifests_unparseable": unparseable,
        "deps_seen": deps_total,
        "deps_resolved": deps_resolved,
        "deps_in_set": deps_in_set,
        "ecosystems": ecosystems,
    }


def main() -> None:
    if not GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    if not REPOS_PARQUET.exists():
        print(f"ERROR: {REPOS_PARQUET} not found", file=sys.stderr)
        sys.exit(1)

    repos_df = pd.read_parquet(REPOS_PARQUET, columns=["full_name", "language", "stargazers_count"])
    in_set = set(repos_df["full_name"])
    print(f"Top-10K loaded: {len(in_set)} repos")

    probe_repos = pick_probe_repos(repos_df)
    print(f"\nProbing {len(probe_repos)} repos:")
    for nwo in probe_repos:
        lang = repos_df.loc[repos_df["full_name"] == nwo, "language"].iloc[0]
        print(f"  [{lang:>10}] {nwo}")

    raw_responses, summaries = [], []

    for full_name in probe_repos:
        owner, name = full_name.split("/", 1)
        print(f"\n→ {full_name}")
        t0 = time.time()
        try:
            body = graphql(DEPENDENCY_QUERY, {"owner": owner, "name": name})
        except RuntimeError as e:
            print(f"  ! gave up: {e}")
            summaries.append({"full_name": full_name, "error": True})
            continue
        elapsed = time.time() - t0

        if "errors" in body:
            for err in body["errors"]:
                print(f"  ! GraphQL error: {err.get('message')}")

        data = body.get("data") or {}
        rate_info = data.get("rateLimit") or {}
        repo_data = data.get("repository")

        raw_responses.append(
            {
                "full_name": full_name,
                "rate_limit": rate_info,
                "repository": repo_data,
                "errors": body.get("errors"),
            }
        )

        if repo_data is None:
            print("  null repository (no access or repo missing)")
            summaries.append({"full_name": full_name, "error": True})
            continue

        s = summarize(repo_data, in_set)
        s.update(
            {
                "full_name": full_name,
                "rate_cost": rate_info.get("cost"),
                "rate_remaining": rate_info.get("remaining"),
                "elapsed_s": round(elapsed, 2),
            }
        )
        summaries.append(s)

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    PROBE_OUTPUT.write_text(json.dumps(raw_responses, indent=2, default=str))
    print(f"\nRaw responses → {PROBE_OUTPUT}")

    print("\n" + "=" * 80)
    print("PER-REPO SUMMARY")
    print("=" * 80)
    for s in summaries:
        if s.get("error"):
            print(f"\n{s['full_name']}: ERROR")
            continue
        print(f"\n{s['full_name']}")
        print(
            f"  manifests: {s['manifests_fetched']} / {s['manifests_total']}  unparseable: {s['manifests_unparseable']}"
        )
        print(f"  deps: seen={s['deps_seen']}  resolved={s['deps_resolved']}  in_set={s['deps_in_set']}")
        if s["deps_seen"]:
            res_pct = 100 * s["deps_resolved"] / s["deps_seen"]
            in_pct = 100 * s["deps_in_set"] / s["deps_seen"]
            print(f"  resolution: {res_pct:.1f}%   in-set: {in_pct:.1f}%")
        eco = sorted(s["ecosystems"].items(), key=lambda kv: -kv[1])
        print(f"  ecosystems: {eco}")
        print(f"  cost: {s['rate_cost']}  remaining: {s['rate_remaining']}  elapsed: {s['elapsed_s']}s")

    valid = [s for s in summaries if not s.get("error")]
    if not valid:
        return

    total_seen = sum(s["deps_seen"] for s in valid)
    total_resolved = sum(s["deps_resolved"] for s in valid)
    total_in_set = sum(s["deps_in_set"] for s in valid)
    total_cost = sum(s["rate_cost"] or 0 for s in valid)
    eco_agg: dict[str, int] = {}
    for s in valid:
        for pm, n in s["ecosystems"].items():
            eco_agg[pm] = eco_agg.get(pm, 0) + n

    print("\n" + "=" * 80)
    print("AGGREGATE")
    print("=" * 80)
    print(f"  repos probed: {len(valid)}")
    print(f"  total deps seen: {total_seen}")
    if total_seen:
        print(f"  resolved-to-repo: {total_resolved} ({100 * total_resolved / total_seen:.1f}%)")
        print(f"  in-set (top-10K): {total_in_set} ({100 * total_in_set / total_seen:.1f}%)")
    print(f"  ecosystems: {sorted(eco_agg.items(), key=lambda kv: -kv[1])}")
    print(f"  total rate-limit cost: {total_cost}")
    elapsed_avg = sum(s["elapsed_s"] for s in valid) / len(valid)
    print(f"  avg query elapsed: {elapsed_avg:.2f}s")

    if total_cost and len(valid):
        per_query_cost = total_cost / len(valid)
        print("\n  Extrapolation for 10K repos (single query each, no batching):")
        print(f"    estimated total cost: {int(per_query_cost * 10_000)} points")
        print("    GitHub limit: 5000 points/hour for personal tokens")


if __name__ == "__main__":
    main()
