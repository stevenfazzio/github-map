"""Tests for config.py constants and paths."""

from pathlib import Path

from pipeline.config import (
    ANTHROPIC_API_KEY,
    CANDIDATES_CSV,
    CO_API_KEY,
    COHERE_BATCH_SIZE,
    COHERE_EMBED_DIMENSION,
    DATA_DIR,
    EMBEDDINGS_NPZ,
    FETCH_OVERSHOOT_COUNT,
    GCP_PROJECT,
    GITHUB_MAP_HTML,
    GITHUB_TOKEN,
    GRAPHQL_BATCH_SIZE,
    LABELS_PARQUET,
    METADATA_PARQUET,
    REPOS_PARQUET,
    TARGET_REPO_COUNT,
    TOPONYMY_MODEL_JOBLIB,
    UMAP_COORDS_NPZ,
)


def test_data_dir_is_path():
    assert isinstance(DATA_DIR, Path)


def test_all_data_paths_under_data_dir():
    data_paths = [
        CANDIDATES_CSV,
        METADATA_PARQUET,
        REPOS_PARQUET,
        EMBEDDINGS_NPZ,
        UMAP_COORDS_NPZ,
        TOPONYMY_MODEL_JOBLIB,
        LABELS_PARQUET,
        GITHUB_MAP_HTML,
    ]
    for p in data_paths:
        assert isinstance(p, Path)
        assert p.parts[0] == "data", f"{p} is not under data/"


def test_constants_have_expected_values():
    assert TARGET_REPO_COUNT == 10_000
    assert FETCH_OVERSHOOT_COUNT == 11_000
    assert GRAPHQL_BATCH_SIZE == 50
    assert COHERE_BATCH_SIZE == 96
    assert COHERE_EMBED_DIMENSION == 512


def test_api_keys_are_strings():
    for key in (GITHUB_TOKEN, CO_API_KEY, ANTHROPIC_API_KEY, GCP_PROJECT):
        assert isinstance(key, str)
