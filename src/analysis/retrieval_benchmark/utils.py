import os
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cosine_similarity_matrix(embeddings):

    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return np.dot(norm, norm.T)
