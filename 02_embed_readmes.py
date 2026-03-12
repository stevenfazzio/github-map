"""Embed READMEs with Cohere embed-v4.0."""

import cohere
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CO_API_KEY, COHERE_BATCH_SIZE, COHERE_EMBED_DIMENSION, EMBEDDINGS_NPZ, REPOS_PARQUET


def main():
    df = pd.read_parquet(REPOS_PARQUET)
    print(f"Loaded {len(df)} repos")

    # Drop repos with READMEs too short to embed meaningfully
    MIN_README_CHARS = 200
    readme_len = df["readme"].str.strip().str.len().fillna(0)
    short_mask = readme_len < MIN_README_CHARS
    dropped = short_mask.sum()
    if dropped:
        print(f"Dropping {dropped} repos with README < {MIN_README_CHARS} chars")
        df = df[~short_mask].reset_index(drop=True)
        df.to_parquet(REPOS_PARQUET, index=False)
        print(f"Saved filtered {len(df)} repos to {REPOS_PARQUET}")

    texts = [row["readme"].strip() for _, row in df.iterrows()]

    co = cohere.ClientV2(api_key=CO_API_KEY)

    all_embeddings = []
    for i in tqdm(range(0, len(texts), COHERE_BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + COHERE_BATCH_SIZE]
        resp = co.embed(
            texts=batch,
            model="embed-v4.0",
            input_type="clustering",
            embedding_types=["float"],
            output_dimension=COHERE_EMBED_DIMENSION,
        )
        all_embeddings.extend(resp.embeddings.float_)

    embeddings = np.array(all_embeddings)
    np.savez(EMBEDDINGS_NPZ, embeddings=embeddings)
    print(f"Saved embeddings {embeddings.shape} to {EMBEDDINGS_NPZ}")


if __name__ == "__main__":
    main()
