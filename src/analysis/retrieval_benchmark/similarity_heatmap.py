import matplotlib.pyplot as plt
import seaborn as sns
from .utils import cosine_similarity_matrix


def plot_similarity_heatmap(embeddings, save_path):

    sim = cosine_similarity_matrix(embeddings)

    plt.figure(figsize=(8,6))

    sns.heatmap(sim, cmap="viridis")

    plt.title("Embedding Similarity Heatmap")

    plt.savefig(save_path, dpi=200)

    plt.close()
