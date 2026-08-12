import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_embedding(embeddings, labels):

    metrics = {}

    # =====================
    # 聚类指标
    # =====================

    metrics["silhouette_score"] = float(
        silhouette_score(embeddings, labels)
    )

    metrics["davies_bouldin_index"] = float(
        davies_bouldin_score(embeddings, labels)
    )

    metrics["calinski_harabasz_score"] = float(
        calinski_harabasz_score(embeddings, labels)
    )

    # =====================
    # Retrieval指标
    # =====================

    retrieval_metrics = topk_retrieval_accuracy(
        embeddings,
        labels,
        topk=(1, 3, 5)
    )

    metrics.update(retrieval_metrics)

    return metrics


def topk_retrieval_accuracy(embeddings, labels, topk=(1, 3, 5)):

    n = len(embeddings)

    # cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    results = {f"top{k}_accuracy": 0 for k in topk}

    for i in range(n):

        # 自己不参与
        sim = sim_matrix[i]
        sim[i] = -1

        # 排序
        ranked = np.argsort(sim)[::-1]

        for k in topk:

            topk_idx = ranked[:k]

            if np.any(labels[topk_idx] == labels[i]):
                results[f"top{k}_accuracy"] += 1

    for k in topk:
        results[f"top{k}_accuracy"] /= n

    return results


def print_metrics(metrics):

    print("\n================ Embedding Quality Metrics ================\n")

    for k, v in metrics.items():

        if "accuracy" in k:
            print(f"{k:30s}: {v*100:.2f}%")
        else:
            print(f"{k:30s}: {v:.4f}")

    print("\n===========================================================\n")
