import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

try:
    from utils import get_next_filename
except ImportError:
    from tools.visualize.utils import get_next_filename


def plot_pca(embeddings, labels, save_dir="fig/pca"):

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

    path = get_next_filename(save_dir, "pca")

    plt.savefig(path)
    plt.close()

    print("Saved:", path)
