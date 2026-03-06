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

    files = glob.glob("data/dxf/*.dxf")

    visualize(files)
