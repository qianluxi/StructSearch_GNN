import os
import random
import numpy as np


def split_dataset(n_samples, train_ratio=0.8, seed=42):
    """按固定随机种子划分训练/验证集，保证结果可复现。

    Args:
        n_samples: 样本总数
        train_ratio: 训练集比例
        seed: 随机种子，默认 42

    Returns:
        train_idx, val_idx: 打乱后的训练/验证索引
    """
    indices = np.arange(n_samples)

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    split = int(n_samples * train_ratio)

    train_idx = indices[:split]
    val_idx = indices[split:]

    return train_idx, val_idx
