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
	uv run python 01b_summarize_readmes.py
	uv run python 02_embed_readmes.py
	uv run python 03_reduce_umap.py
	uv run python 04_label_topics.py
	uv run python 05_visualize.py

clean:
	@echo "This will remove all files in data/. Press Ctrl+C to cancel."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/*
