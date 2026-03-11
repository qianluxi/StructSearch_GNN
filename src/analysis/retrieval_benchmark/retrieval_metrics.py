import numpy as np
import matplotlib.pyplot as plt
from .utils import cosine_similarity_matrix


def compute_topk_accuracy(embeddings, labels, max_k=10):

    sim = cosine_similarity_matrix(embeddings)

    n = len(labels)

    acc = []

    for k in range(1, max_k + 1):

        correct = 0

        for i in range(n):

            idx = np.argsort(-sim[i])

            idx = idx[1:k+1]

            if np.any(labels[idx] == labels[i]):
                correct += 1

        acc.append(correct / n)

    return acc


def plot_accuracy_curve(acc, save_path):

    k = list(range(1, len(acc)+1))

    plt.figure()

    plt.plot(k, acc, marker="o")

    plt.xlabel("Top-K")
    plt.ylabel("Accuracy")

    plt.title("Retrieval Accuracy Curve")

    plt.grid(True)

    plt.savefig(save_path, dpi=200)

    plt.close()
