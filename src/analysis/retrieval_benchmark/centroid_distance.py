import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def plot_centroid_distance(embeddings, labels, save_path):

    classes = np.unique(labels)

    centroids = []

    for c in classes:

        centroids.append(embeddings[labels==c].mean(axis=0))

    centroids = np.vstack(centroids)

    sim = cosine_similarity(centroids)

    plt.figure()

    sns.heatmap(sim, annot=True, cmap="coolwarm")

    plt.title("Class Centroid Cosine Similarity")

    plt.savefig(save_path, dpi=200)

    plt.close()
