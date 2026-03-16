"""Tests for pure functions in 01_fetch_repos.py."""

import importlib
import sys
from pathlib import Path

# Ensure pipeline/ is on sys.path so `from config import ...` works inside the script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

# Import the module with its dotted-number name
spec = importlib.util.spec_from_file_location("fetch_repos", "pipeline/01_fetch_repos.py")
fetch_repos = importlib.util.module_from_spec(spec)
sys.modules["fetch_repos"] = fetch_repos
spec.loader.exec_module(fetch_repos)

_extract_readme = fetch_repos._extract_readme
_parse_metadata = fetch_repos._parse_metadata
_build_batch_query = fetch_repos._build_batch_query
_build_readme_fragment = fetch_repos._build_readme_fragment
README_ALIASES = fetch_repos.README_ALIASES
METADATA_FRAGMENT = fetch_repos.METADATA_FRAGMENT
README_FRAGMENT = fetch_repos.README_FRAGMENT


class TestExtractReadme:
    def test_picks_first_non_null_alias(self):
        node = {
            "readme_md": None,
            "readme_lower": {"text": "hello from readme.md"},
            "readme_rst": {"text": "rst content"},
        }
        assert _extract_readme(node) == "hello from readme.md"

    def test_returns_empty_when_all_null(self):
        node = {alias: None for alias, _ in README_ALIASES}
        assert _extract_readme(node) == ""

    def test_skips_empty_text(self):
        node = {
            "readme_md": {"text": ""},
            "readme_lower": {"text": "actual content"},
        }
        assert _extract_readme(node) == "actual content"

    def test_returns_empty_for_empty_node(self):
        assert _extract_readme({}) == ""


class TestParseMetadata:
    def _make_full_node(self):
        return {
            "nameWithOwner": "octocat/hello-world",
            "description": "A test repo",
            "primaryLanguage": {"name": "Python"},
            "stargazerCount": 42000,
            "licenseInfo": {"spdxId": "MIT"},
            "createdAt": "2020-01-01T00:00:00Z",
            "repositoryTopics": {"nodes": [{"topic": {"name": "python"}}, {"topic": {"name": "testing"}}]},
            "pushedAt": "2024-06-01T00:00:00Z",
            "forkCount": 1500,
            "isArchived": False,
            "diskUsage": 5000,
            "hasWikiEnabled": True,
            "hasDiscussionsEnabled": False,
            "watchers": {"totalCount": 100},
            "issues": {"totalCount": 50},
            "pullRequests": {"totalCount": 10},
            "releases": {"totalCount": 5},
            "discussions": {"totalCount": 3},
            "fundingLinks": [{"platform": "GITHUB", "url": "https://github.com/sponsors/octocat"}],
            "defaultBranchRef": {
                "name": "main",
                "target": {"history": {"totalCount": 999}},
            },
            "owner": {"__typename": "Organization"},
            "languages": {
                "edges": [
                    {"size": 10000, "node": {"name": "Python"}},
                    {"size": 2000, "node": {"name": "Shell"}},
                ]
            },
        }

    def test_parses_full_node(self):
        row = _parse_metadata(self._make_full_node())
        assert row["full_name"] == "octocat/hello-world"
        assert row["description"] == "A test repo"
        assert row["language"] == "Python"
        assert row["stargazers_count"] == 42000
        assert row["license"] == "MIT"
        assert row["topics"] == "python,testing"
        assert row["fork_count"] == 1500
        assert row["is_archived"] is False
        assert row["commit_count"] == 999
        assert row["owner_type"] == "Organization"
        assert row["has_funding"] is True
        assert row["default_branch"] == "main"

    def test_handles_missing_optional_fields(self):
        node = {
            "nameWithOwner": "user/repo",
            "stargazerCount": 10,
            "primaryLanguage": None,
            "licenseInfo": None,
            "repositoryTopics": None,
            "defaultBranchRef": None,
            "owner": None,
            "languages": None,
        }
        row = _parse_metadata(node)
        assert row["full_name"] == "user/repo"
        assert row["language"] == ""
        assert row["license"] == ""
        assert row["topics"] == ""
        assert row["default_branch"] == ""
        assert row["commit_count"] == 0
        assert row["owner_type"] == ""


class TestBuildBatchQuery:
    def test_produces_valid_query_with_metadata(self):
        repos = ["octocat/hello-world", "torvalds/linux"]
        query = _build_batch_query(repos, METADATA_FRAGMENT)
        assert "repo0: repository(" in query
        assert "repo1: repository(" in query
        assert "...MetadataFields" in query
        assert 'owner: "octocat"' in query
        assert 'name: "hello-world"' in query
        assert 'owner: "torvalds"' in query

    def test_produces_valid_query_with_readme(self):
        repos = ["user/repo"]
        query = _build_batch_query(repos, README_FRAGMENT)
        assert "...ReadmeFields" in query

    def test_empty_batch(self):
        query = _build_batch_query([], METADATA_FRAGMENT)
        assert "query {" in query


class TestBuildReadmeFragment:
    def test_contains_all_aliases(self):
        fragment = _build_readme_fragment()
        for alias, filename in README_ALIASES:
            assert alias in fragment
            assert filename in fragment

    def test_is_valid_graphql_fragment(self):
        fragment = _build_readme_fragment()
        assert fragment.startswith("fragment ReadmeFields on Repository {")
        assert fragment.endswith("}")
