import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import cosine_similarity_matrix


def visualize_cases(embeddings, labels, save_dir, top_k=5, n_cases=5):

    sim = cosine_similarity_matrix(embeddings)

    n = len(labels)

    queries = np.random.choice(n, n_cases, replace=False)

    for q in queries:

        idx = np.argsort(-sim[q])[1:top_k+1]

        fig, ax = plt.subplots()

        ax.bar(range(top_k), sim[q][idx])

        ax.set_title(f"Query {q} label={labels[q]}")

        path = os.path.join(save_dir, f"query_{q}.png")

        plt.savefig(path, dpi=200)

        plt.close()
