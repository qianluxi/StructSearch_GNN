import numpy as np
import matplotlib.pyplot as plt


def plot_variance(embeddings, save_path):

    var = embeddings.var(axis=0)

    plt.figure()

    plt.hist(var, bins=20)

    plt.title("Embedding Variance Distribution")

    plt.xlabel("Variance")

    plt.ylabel("Count")

    plt.savefig(save_path, dpi=200)

    plt.close()
