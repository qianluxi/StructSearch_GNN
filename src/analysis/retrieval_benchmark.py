import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

from sklearn.metrics.pairwise import cosine_similarity

from tools.visualize.embedding_extractor import extract_embeddings


def retrieval_benchmark(topk=5):

    embeddings, labels, preds = extract_embeddings()

    n = len(embeddings)

    sim_matrix = cosine_similarity(embeddings)

    top1_correct = 0
    topk_correct = 0

    print("\n================ Retrieval Benchmark ================\n")

    for i in range(n):

        sim = sim_matrix[i]
        sim[i] = -1

        ranked = np.argsort(sim)[::-1]

        topk_idx = ranked[:topk]

        query_label = labels[i]

        top_labels = labels[topk_idx]

        if top_labels[0] == query_label:
            top1_correct += 1

        if np.any(top_labels == query_label):
            topk_correct += 1

        print(f"Query {i:02d} | label={query_label}")

        for r, idx in enumerate(topk_idx):

            match = "✓" if labels[idx] == query_label else " "

            print(
                f"  {r+1}. id={idx:02d} "
                f"label={labels[idx]} "
                f"sim={sim_matrix[i][idx]:.3f} {match}"
            )

        print()

    top1_acc = top1_correct / n
    topk_acc = topk_correct / n

    print("===================================================")

    print(f"Top1 Accuracy : {top1_acc*100:.2f}%")
    print(f"Top{topk} Accuracy : {topk_acc*100:.2f}%")

    print("===================================================")


if __name__ == "__main__":

    retrieval_benchmark(topk=5)
