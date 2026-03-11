import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(embeddings, labels, save_path):

    tsne = TSNE(n_components=2, random_state=42)

    emb2d = tsne.fit_transform(embeddings)

    plt.figure()

    plt.scatter(
        emb2d[:,0],
        emb2d[:,1],
        c=labels,
        cmap="tab10"
    )

    plt.title("Embedding t-SNE")

    plt.savefig(save_path, dpi=200)

    plt.close()
