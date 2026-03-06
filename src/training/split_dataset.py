import os
import random


def split_dataset(dxf_dir, train_ratio=0.8):

    files = [
        os.path.join(dxf_dir, f)
        for f in os.listdir(dxf_dir)
        if f.endswith(".dxf")
    ]

    random.shuffle(files)

    split = int(len(files) * train_ratio)

    train_files = files[:split]
    val_files = files[split:]

    return train_files, val_files
