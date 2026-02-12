import os
import numpy as np
import src.config as cfg


def load_feature_index():

    index_dir = cfg.EMBEDDING_DIR

    embeddings_path = os.path.join(index_dir, "embeddings.npy")
    filenames_path = os.path.join(index_dir, "filenames.npy")
    labels_path = os.path.join(index_dir, "labels.npy")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            "未找到 embedding 索引，请先运行 index_builder.py"
        )

    embeddings = np.load(embeddings_path)
    filenames = np.load(filenames_path)
    labels = np.load(labels_path)

    return embeddings, filenames, labels
