"""Shared paths, constants, and env var loading."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load centralized credentials, then project-local overrides
load_dotenv(Path.home() / ".config" / "data-apis" / ".env")
load_dotenv(override=True)

# ── Directories ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── File paths ───────────────────────────────────────────────────────────────
CANDIDATES_CSV = DATA_DIR / "candidates.csv"
CANDIDATES_COMMITTED = Path("candidates.csv")
METADATA_PARQUET = DATA_DIR / "metadata.parquet"
REPOS_PARQUET = DATA_DIR / "repos.parquet"
EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"
UMAP_COORDS_NPZ = DATA_DIR / "umap_coords.npz"
TOPONYMY_MODEL_JOBLIB = DATA_DIR / "toponymy_model.joblib"
LABELS_PARQUET = DATA_DIR / "labels.parquet"
GITHUB_MAP_HTML = DATA_DIR / "github_map.html"
METHODOLOGY_HTML = DATA_DIR / "methodology.html"

# ── Docs (GitHub Pages) ──────────────────────────────────────────────────────
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)
DOCS_INDEX_HTML = DOCS_DIR / "index.html"
DOCS_METHODOLOGY_HTML = DOCS_DIR / "methodology.html"

# ── API keys ─────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
CO_API_KEY = os.environ.get("CO_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "")

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_REPO_COUNT = 10_000
FETCH_OVERSHOOT_COUNT = 11_000
GRAPHQL_BATCH_SIZE = 50
COHERE_BATCH_SIZE = 96
COHERE_EMBED_DIMENSION = 512
ANTHROPIC_MODEL_SUMMARIZE = "claude-haiku-4-5"

# ── Experiments ─────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = DATA_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
