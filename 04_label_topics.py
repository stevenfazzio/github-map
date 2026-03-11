"""Generate topic labels using Toponymy with Claude LLM."""

import asyncio
import joblib
import numpy as np
import pandas as pd
from toponymy import Toponymy, ToponymyClusterer
from toponymy.llm_wrappers import AsyncAnthropicNamer
from sentence_transformers import SentenceTransformer

# Workaround: nested asyncio.run() calls fail with "Event loop is closed".
# nest_asyncio patches the loop to allow re-entrant calls.
import nest_asyncio
nest_asyncio.apply()

from config import (
    ANTHROPIC_API_KEY,
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

    # Truncate READMEs to ~8K chars (~2K tokens) to stay within LLM context limits
    # when Toponymy batches multiple exemplars into a single prompt
    MAX_README_CHARS = 2_000

    readmes = []
    for _, row in df.iterrows():
        text = row["readme"].strip() if isinstance(row["readme"], str) else ""
        if not text:
            text = row["description"].strip() if isinstance(row["description"], str) else ""
        if not text:
            text = row["full_name"]
        readmes.append(text[:MAX_README_CHARS])

    print(f"Loaded {len(readmes)} documents")

    # Set up Toponymy components
    llm = AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model="claude-sonnet-4-20250514")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    clusterer = ToponymyClusterer(min_clusters=4, max_layers=2)

    # Fit clusterer
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    # Fit Toponymy model
    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="GitHub repository READMEs",
        corpus_description="collection of the top 1,000 most-starred GitHub repositories",
        exemplar_delimiters=['    * """', '"""\n'],
    )
    topic_model.fit(
        objects=readmes,
        embedding_vectors=embeddings,
        clusterable_vectors=coords,
    )

    # Extract per-document labels from cluster layers
    n_layers = len(topic_model.cluster_layers_)
    coarse_labels = []
    fine_labels = []

    if n_layers >= 2:
        coarse_layer = topic_model.cluster_layers_[0]
        fine_layer = topic_model.cluster_layers_[-1]
    elif n_layers == 1:
        coarse_layer = topic_model.cluster_layers_[0]
        fine_layer = topic_model.cluster_layers_[0]
    else:
        raise ValueError("No cluster layers found")

    for i in range(len(readmes)):
        coarse_labels.append(coarse_layer.topic_name_vector[i])
        fine_labels.append(fine_layer.topic_name_vector[i])

    # Save labels
    labels_df = pd.DataFrame(
        {
            "full_name": df["full_name"],
            "coarse_label": coarse_labels,
            "fine_label": fine_labels,
        }
    )
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
