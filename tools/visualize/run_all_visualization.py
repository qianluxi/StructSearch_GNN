from embedding_extractor import extract_embeddings
from tsne_plot import plot_tsne
from pca_plot import plot_pca
from distance_matrix import plot_distance_matrix
from confusion_matrix_plot import plot_confusion_matrix


def main():

    embeddings, labels, preds = extract_embeddings()

    plot_tsne(embeddings, labels)

    plot_pca(embeddings, labels)

    plot_distance_matrix(embeddings)

    plot_confusion_matrix(labels, preds)


if __name__ == "__main__":
    main()
