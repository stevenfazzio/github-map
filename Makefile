.PHONY: install lint format test pipeline clean

install:
	uv sync --extra dev

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

pipeline:
	uv run python 00_enumerate_repos.py
	uv run python 01_fetch_repos.py
	uv run python 02_select_top_repos.py
	uv run python 03_summarize_readmes.py
	uv run python 04_embed_readmes.py
	uv run python 05_reduce_umap.py
	uv run python 06_label_topics.py
	uv run python 07_visualize.py

clean:
	@echo "This will remove all files in data/. Press Ctrl+C to cancel."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/*
