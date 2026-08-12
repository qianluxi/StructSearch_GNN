import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

from tools.visualize.embedding_extractor import extract_embeddings


def check_embedding_difference(embeddings):

    print("\n=========== Check Embedding Difference ===========\n")

    print("||emb0 - emb1|| =", np.linalg.norm(embeddings[0] - embeddings[1]))
    print("||emb0 - emb22|| =", np.linalg.norm(embeddings[0] - embeddings[22]))
    print("||emb0 - emb30|| =", np.linalg.norm(embeddings[0] - embeddings[30]))

    print("\n(如果不是0，说明embedding并不完全相同)\n")


def check_embedding_variance(embeddings):

    print("\n=========== Embedding Variance ===========\n")

    var_per_dim = np.var(embeddings, axis=0)

    print("Mean variance :", np.mean(var_per_dim))
    print("Max variance  :", np.max(var_per_dim))
    print("Min variance  :", np.min(var_per_dim))

    print("\nVariance per dimension (first 10):")
    print(var_per_dim[:10])


def compute_class_centroids(embeddings, labels):

    print("\n=========== Class Centroid Distance ===========\n")

    classes = np.unique(labels)

    centroids = {}

    for c in classes:
        centroids[c] = embeddings[labels == c].mean(axis=0)

    for c in classes:
        print(f"Class {c} centroid (first 5 dims):", centroids[c][:5])

    print("\nCentroid cosine similarity matrix:\n")

    centroid_matrix = np.array([centroids[c] for c in classes])

    sim = cosine_similarity(centroid_matrix)

    print(sim)


def plot_structure_embedding_map(embeddings, labels):

    print("\n=========== Plot Structure Embedding Map ===========\n")

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(6,6))

    colors = {
        0: "red",
        1: "blue",
        2: "green"
    }

    names = {
        0: "Shear Wall",
        1: "Frame-Shear",
        2: "Frame"
    }

    for c in np.unique(labels):

        idx = labels == c

        plt.scatter(
            emb_2d[idx,0],
            emb_2d[idx,1],
            c=colors[c],
            label=names[c],
            s=60
        )

    plt.title("Structure Embedding Map (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)

    save_dir = os.path.join(ROOT_DIR, "fig", "embedding_map")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "structure_embedding_map.png")

    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Map saved to:", save_path)


def run_diagnostics():

    embeddings, labels, preds = extract_embeddings()

    check_embedding_difference(embeddings)

    check_embedding_variance(embeddings)

    compute_class_centroids(embeddings, labels)

    plot_structure_embedding_map(embeddings, labels)


if __name__ == "__main__":

    run_diagnostics()
