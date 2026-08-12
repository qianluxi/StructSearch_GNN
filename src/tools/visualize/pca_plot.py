import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from utils import get_next_filename


def plot_pca(embeddings, labels):

    pca = PCA(n_components=2)

    emb_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(8,6))

    for c in np.unique(labels):

        idx = labels == c

        plt.scatter(
            emb_2d[idx,0],
            emb_2d[idx,1],
            label=f"class {c}",
            s=60
        )

    plt.legend()

    plt.title("Graph Embedding PCA")

    path = get_next_filename("fig/pca", "pca")

    plt.savefig(path)
    plt.close()

    print("Saved:", path)
