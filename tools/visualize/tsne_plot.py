import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from utils import get_next_filename


def plot_tsne(embeddings, labels):

    tsne = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate=200,
        random_state=42
    )

    emb_2d = tsne.fit_transform(embeddings)

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
    plt.title("Graph Embedding TSNE")

    path = get_next_filename("fig/tsne", "tsne")

    plt.savefig(path)
    plt.close()

    print("Saved:", path)
