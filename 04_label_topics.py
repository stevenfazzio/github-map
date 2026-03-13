"""Generate topic labels using Toponymy with Claude LLM."""

import asyncio
import joblib
import numpy as np
import pandas as pd
from toponymy import Toponymy, ToponymyClusterer
from toponymy.llm_wrappers import AsyncAnthropicNamer
from toponymy.embedding_wrappers import CohereEmbedder

# Workaround: nested asyncio.run() calls fail with "Event loop is closed".
# nest_asyncio patches the loop to allow re-entrant calls.
import nest_asyncio
nest_asyncio.apply()

from config import (
    ANTHROPIC_API_KEY,
    CO_API_KEY,
    EMBEDDINGS_NPZ,
    LABELS_PARQUET,
    REPOS_PARQUET,
    TOPONYMY_MODEL_JOBLIB,
    UMAP_COORDS_NPZ,
)


def main():
    # Load data
    df = pd.read_parquet(REPOS_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    coords = np.load(UMAP_COORDS_NPZ)["coords"]

    # Use LLM-generated summaries if available (from 01b_summarize_readmes.py),
    # falling back to truncated READMEs
    MAX_README_CHARS = 2_000
    has_summary = "summary" in df.columns

    documents = []
    for _, row in df.iterrows():
        text = ""
        if has_summary and isinstance(row.get("summary"), str):
            text = row["summary"].strip()
        if not text:
            text = row["readme"].strip() if isinstance(row["readme"], str) else ""
            text = text[:MAX_README_CHARS]
        if not text:
            text = row["description"].strip() if isinstance(row["description"], str) else ""
        if not text:
            text = row["full_name"]
        documents.append(text)

    print(f"Loaded {len(documents)} documents")

    # Set up Toponymy components
    llm = AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model="claude-sonnet-4-20250514")
    embedder = CohereEmbedder(api_key=CO_API_KEY, model="embed-v4.0")
    clusterer = ToponymyClusterer(min_clusters=4)

    # Fit clusterer
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    # Fit Toponymy model
    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="GitHub repository descriptions",
        corpus_description="collection of the top 1,000 most-starred GitHub repositories",
        exemplar_delimiters=['    * """', '"""\n'],
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=coords,
    )

    # Extract per-document labels from all cluster layers
    # Layer 0 is finest, last layer is coarsest — DataMapPlot expects coarsest first
    n_layers = len(topic_model.cluster_layers_)
    if n_layers == 0:
        raise ValueError("No cluster layers found")
    print(f"Toponymy produced {n_layers} cluster layer(s)")

    labels_dict = {"full_name": df["full_name"]}
    for i, layer in enumerate(reversed(topic_model.cluster_layers_)):
        labels_dict[f"label_layer_{i}"] = layer.topic_name_vector

    labels_df = pd.DataFrame(labels_dict)
    labels_df.to_parquet(LABELS_PARQUET, index=False)
    print(f"Saved labels to {LABELS_PARQUET}")

    # Save model (skip if unpicklable due to async client locks)
    try:
        joblib.dump(topic_model, TOPONYMY_MODEL_JOBLIB)
        print(f"Saved model to {TOPONYMY_MODEL_JOBLIB}")
    except TypeError:
        print("Skipped saving model (async client is not picklable)")


if __name__ == "__main__":
    main()
