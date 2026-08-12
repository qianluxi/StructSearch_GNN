import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

try:
    from utils import get_next_filename
except ImportError:
    from tools.visualize.utils import get_next_filename


def plot_distance_matrix(embeddings, save_dir="fig/distance"):

    dist = euclidean_distances(embeddings)

    plt.figure(figsize=(8,6))

    plt.imshow(dist, cmap="viridis")

    plt.colorbar()

    plt.title("Embedding Distance Matrix")

    path = get_next_filename(save_dir, "distance")

    plt.savefig(path)
    plt.close()

    print("Saved:", path)
