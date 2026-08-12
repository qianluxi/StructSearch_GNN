import os
import random
import numpy as np


def split_dataset(n_samples, train_ratio=0.8):
    
    indices = np.arange(n_samples)

    np.random.shuffle(indices)

    split = int(n_samples * train_ratio)

    train_idx = indices[:split]
    val_idx = indices[split:]

    return train_idx, val_idx
