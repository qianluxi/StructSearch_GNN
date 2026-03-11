import os

from .embedding_extractor import extract_embeddings
from .retrieval_metrics import compute_topk_accuracy, plot_accuracy_curve
from .similarity_heatmap import plot_similarity_heatmap
from .embedding_tsne import plot_tsne
from .centroid_distance import plot_centroid_distance
from .embedding_variance import plot_variance
from .retrieval_cases import visualize_cases
from .utils import ensure_dir


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def run():

    fig_dir = os.path.join(ROOT_DIR, "fig", "benchmark")

    ensure_dir(fig_dir)

    case_dir = os.path.join(fig_dir, "retrieval_cases")

    ensure_dir(case_dir)

    embeddings, labels = extract_embeddings()

    print("\nRunning benchmark...\n")

    # TSNE
    plot_tsne(
        embeddings,
        labels,
        os.path.join(fig_dir, "embedding_tsne.png")
    )

    # Similarity heatmap
    plot_similarity_heatmap(
        embeddings,
        os.path.join(fig_dir, "similarity_heatmap.png")
    )

    # centroid distance
    plot_centroid_distance(
        embeddings,
        labels,
        os.path.join(fig_dir, "centroid_distance.png")
    )

    # variance
    plot_variance(
        embeddings,
        os.path.join(fig_dir, "embedding_variance.png")
    )

    # retrieval accuracy
    acc = compute_topk_accuracy(embeddings, labels)

    plot_accuracy_curve(
        acc,
        os.path.join(fig_dir, "retrieval_accuracy_curve.png")
    )

    # retrieval cases
    visualize_cases(
        embeddings,
        labels,
        case_dir
    )

    print("\nBenchmark finished")
    print("Results saved to:", fig_dir)


if __name__ == "__main__":
    run()
