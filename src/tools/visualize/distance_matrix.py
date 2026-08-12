import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

from utils import get_next_filename


def plot_distance_matrix(embeddings):

    dist = euclidean_distances(embeddings)

    plt.figure(figsize=(8,6))

    plt.imshow(dist, cmap="viridis")

    plt.colorbar()

    plt.title("Embedding Distance Matrix")

    path = get_next_filename("fig/distance", "distance")

    plt.savefig(path)
    plt.close()

    print("Saved:", path)
