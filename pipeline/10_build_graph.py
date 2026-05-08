"""Assemble unified nodes + edges parquet from raw dependency data.

Inputs:
  data/repos.parquet           (top-10K source repos, from stage 02/03)
  data/external_repos.parquet  (out-of-set GitHub targets, from stage 09)
  data/dependencies.parquet    (raw edges, from stage 08)

Outputs:
  data/nodes.parquet  — one row per node, types: top10k_repo, external_repo,
                         external_package; common metadata columns where
                         available (NaN otherwise) plus in_degree, out_degree.
  data/edges.parquet  — one row per dep occurrence, edges keyed by node_id.

Node ID scheme:
  - GitHub repos: "gh:owner/name"
  - Unresolved packages: "pkg:{package_manager}:{package_name}"
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
from config import DATA_DIR, REPOS_PARQUET

DEPS_PARQUET = DATA_DIR / "dependencies.parquet"
EXTERNAL_PARQUET = DATA_DIR / "external_repos.parquet"
NODES_PARQUET = DATA_DIR / "nodes.parquet"
EDGES_PARQUET = DATA_DIR / "edges.parquet"

NODE_COLS = [
    "node_id",
    "type",
    "nwo",
    "package_name",
    "package_manager",
    "description",
    "primary_language",
    "stargazers_count",
    "fork_count",
    "license",
    "created_at",
    "pushed_at",
    "is_archived",
]


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


def _build_in_set_nodes(repos: pd.DataFrame) -> pd.DataFrame:
    """One row per top-10K repo, with metadata mapped onto NODE_COLS."""
    nodes = pd.DataFrame(
        {
            "node_id": "gh:" + repos["full_name"],
            "type": "top10k_repo",
            "nwo": repos["full_name"],
            "package_name": pd.NA,
            "package_manager": pd.NA,
            "description": repos.get("description", ""),
            "primary_language": repos.get("language", ""),
            "stargazers_count": repos.get("stargazers_count", 0),
            "fork_count": repos.get("fork_count", 0),
            "license": repos.get("license", ""),
            "created_at": repos.get("created_at", ""),
            "pushed_at": repos.get("pushed_at", ""),
            "is_archived": repos.get("is_archived", False),
        }
    )
    return nodes[NODE_COLS]


def _build_external_repo_nodes(external: pd.DataFrame) -> pd.DataFrame:
    if external.empty:
        return pd.DataFrame(columns=NODE_COLS)
    nodes = pd.DataFrame(
        {
            "node_id": "gh:" + external["full_name"],
            "type": "external_repo",
            "nwo": external["full_name"],
            "package_name": pd.NA,
            "package_manager": pd.NA,
            "description": external.get("description", ""),
            "primary_language": external.get("primary_language", ""),
            "stargazers_count": external.get("stargazers_count", 0),
            "fork_count": external.get("fork_count", 0),
            "license": external.get("license", ""),
            "created_at": external.get("created_at", ""),
            "pushed_at": external.get("pushed_at", ""),
            "is_archived": external.get("is_archived", False),
        }
    )
    return nodes[NODE_COLS]


def _build_external_package_nodes(edges_raw: pd.DataFrame) -> pd.DataFrame:
    """Distinct (package_manager, package_name) pairs for unresolved targets."""
    unresolved = edges_raw[edges_raw["target_kind"] == "unresolved"]
    if unresolved.empty:
        return pd.DataFrame(columns=NODE_COLS)

    distinct = unresolved[["package_manager", "package_name"]].drop_duplicates().reset_index(drop=True)
    nodes = pd.DataFrame(
        {
            "node_id": "pkg:" + distinct["package_manager"] + ":" + distinct["package_name"],
            "type": "external_package",
            "nwo": pd.NA,
            "package_name": distinct["package_name"],
            "package_manager": distinct["package_manager"],
            "description": "",
            "primary_language": "",
            "stargazers_count": pd.NA,
            "fork_count": pd.NA,
            "license": "",
            "created_at": "",
            "pushed_at": "",
            "is_archived": pd.NA,
        }
    )
    return nodes[NODE_COLS]


def _make_target_id(row: pd.Series) -> str:
    if row["target_kind"] == "unresolved":
        return f"pkg:{row['package_manager']}:{row['package_name']}"
    return f"gh:{row['target_nwo']}"


def _build_edges(edges_raw: pd.DataFrame) -> pd.DataFrame:
    if edges_raw.empty:
        return pd.DataFrame(
            columns=[
                "source_id",
                "target_id",
                "target_kind",
                "package_name",
                "package_manager",
                "manifest_filename",
                "manifest_path",
                "manifest_parseable",
                "manifest_total_deps",
                "requirements",
                "has_dependencies",
            ]
        )
    out = edges_raw.copy()
    out["source_id"] = "gh:" + out["source_nwo"]
    # Vectorized target_id construction (apply is slow on 500K+ rows)
    is_unresolved = out["target_kind"] == "unresolved"
    out["target_id"] = "gh:" + out["target_nwo"].astype("string")
    out.loc[is_unresolved, "target_id"] = (
        "pkg:" + out.loc[is_unresolved, "package_manager"] + ":" + out.loc[is_unresolved, "package_name"]
    )
    return out[
        [
            "source_id",
            "target_id",
            "target_kind",
            "package_name",
            "package_manager",
            "manifest_filename",
            "manifest_path",
            "manifest_parseable",
            "manifest_total_deps",
            "requirements",
            "has_dependencies",
        ]
    ]


def _attach_degrees(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    in_deg = edges.groupby("target_id").size().rename("in_degree")
    out_deg = edges.groupby("source_id").size().rename("out_degree")
    nodes = nodes.merge(in_deg, left_on="node_id", right_index=True, how="left")
    nodes = nodes.merge(out_deg, left_on="node_id", right_index=True, how="left")
    nodes["in_degree"] = nodes["in_degree"].fillna(0).astype(int)
    nodes["out_degree"] = nodes["out_degree"].fillna(0).astype(int)
    return nodes


def main():
    if not DEPS_PARQUET.exists():
        raise SystemExit(f"{DEPS_PARQUET} not found — run stage 08 first")
    if not REPOS_PARQUET.exists():
        raise SystemExit(f"{REPOS_PARQUET} not found")

    print("Loading inputs...")
    repos = pd.read_parquet(REPOS_PARQUET)
    edges_raw = pd.read_parquet(DEPS_PARQUET)
    external = pd.read_parquet(EXTERNAL_PARQUET) if EXTERNAL_PARQUET.exists() else pd.DataFrame()
    print(f"  top-10K repos: {len(repos)}")
    print(f"  external repos enriched: {len(external)}")
    print(f"  raw edges: {len(edges_raw)}")

    if external.empty:
        print("  ! external_repos.parquet missing or empty — running without enrichment")

    print("Building node frames...")
    in_set_nodes = _build_in_set_nodes(repos)
    external_nodes = _build_external_repo_nodes(external)
    package_nodes = _build_external_package_nodes(edges_raw)
    nodes = pd.concat([in_set_nodes, external_nodes, package_nodes], ignore_index=True)
    nodes = nodes.drop_duplicates(subset="node_id", keep="first")
    print(f"  total nodes: {len(nodes)}")
    print(nodes["type"].value_counts().to_string())

    print("Building edges...")
    edges = _build_edges(edges_raw)
    print(f"  total edges: {len(edges)}")

    # Drop edges whose source isn't in the node set (defensive — shouldn't
    # happen, but stage 08 might be writing concurrently with stage 10).
    known_node_ids = set(nodes["node_id"])
    bad_source = ~edges["source_id"].isin(known_node_ids)
    if bad_source.any():
        print(f"  ! dropping {bad_source.sum()} edges with unknown source_id")
        edges = edges[~bad_source]

    # Edges can legitimately point at targets we haven't materialized as nodes
    # yet (e.g., external_repo target whose enrichment row is missing). Add a
    # placeholder node for each such target so the graph is closed.
    missing_targets = sorted(set(edges["target_id"]) - known_node_ids)
    if missing_targets:
        print(f"  ! {len(missing_targets)} edge targets lack node rows — adding placeholders")
        placeholders = pd.DataFrame({"node_id": missing_targets})
        placeholders["type"] = (
            placeholders["node_id"].str.startswith("pkg:").map({True: "external_package", False: "external_repo"})
        )
        for col in NODE_COLS:
            if col not in placeholders.columns:
                placeholders[col] = pd.NA
        # For external_repo placeholders, derive nwo from id
        is_repo = placeholders["type"] == "external_repo"
        placeholders.loc[is_repo, "nwo"] = placeholders.loc[is_repo, "node_id"].str[3:]
        # For external_package placeholders, derive name + manager from id
        is_pkg = placeholders["type"] == "external_package"
        if is_pkg.any():
            split = placeholders.loc[is_pkg, "node_id"].str[4:].str.split(":", n=1, expand=True)
            placeholders.loc[is_pkg, "package_manager"] = split[0]
            placeholders.loc[is_pkg, "package_name"] = split[1]
        nodes = pd.concat([nodes, placeholders[NODE_COLS]], ignore_index=True)

    print("Computing degrees...")
    nodes = _attach_degrees(nodes, edges)

    print(f"\nWriting {NODES_PARQUET} and {EDGES_PARQUET}")
    _safe_write_parquet(nodes, NODES_PARQUET)
    _safe_write_parquet(edges, EDGES_PARQUET)

    print("\nDone.")
    print(f"  nodes: {len(nodes)}")
    print(nodes["type"].value_counts().to_string())
    print(f"\n  edges: {len(edges)}")
    print(edges["target_kind"].value_counts().to_string())
    print("\nTop nodes by in-degree (most depended on):")
    top_in = nodes.sort_values("in_degree", ascending=False).head(15)
    print(top_in[["node_id", "type", "in_degree"]].to_string(index=False))


if __name__ == "__main__":
    main()
