"""Reduce embeddings to 2D with PCA pre-reduction then UMAP."""

import numpy as np
import umap
from sklearn.decomposition import PCA

from config import EMBEDDINGS_NPZ, PCA_DIMS, UMAP_COORDS_NPZ


def main():
    data = np.load(EMBEDDINGS_NPZ)
    embeddings = data["embeddings"]
    print(f"Loaded embeddings: {embeddings.shape}")

    pca = PCA(n_components=PCA_DIMS, random_state=42)
    reduced = pca.fit_transform(embeddings)
    variance = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"PCA {embeddings.shape[1]}d → {PCA_DIMS}d: {variance:.1%} variance retained")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(reduced)

    np.savez(UMAP_COORDS_NPZ, coords=coords)
    print(f"Saved 2D coords {coords.shape} to {UMAP_COORDS_NPZ}")


if __name__ == "__main__":
    main()
