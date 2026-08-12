import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from training.evaluate import extract_embeddings


def search(query_index, embeddings, files, topk=5):

    sim = cosine_similarity(
        embeddings[query_index].reshape(1,-1),
        embeddings
    )[0]

    idx = np.argsort(-sim)[1:topk+1]

    for i in idx:

        print(files[i], sim[i])


if __name__ == "__main__":

    import glob

    files = glob.glob(os.path.join(ROOT_DIR, "data", "raw", "*.dxf"))

    emb = extract_embeddings(files)

    search(0, emb, files)
