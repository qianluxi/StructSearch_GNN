import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from training.evaluate import extract_embeddings


def visualize(dxf_list):

    emb = extract_embeddings(dxf_list)

    tsne = TSNE(n_components=2)

    X = tsne.fit_transform(emb)

    plt.scatter(X[:,0], X[:,1])

    plt.title("Structure Embedding")

    plt.show()


if __name__ == "__main__":

    import glob

    files = glob.glob(os.path.join(ROOT_DIR, "data", "raw", "*.dxf"))

    visualize(files)
