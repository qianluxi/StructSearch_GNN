"""主入口：训练（委托给 src/training/train.py）。"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.training.train import train

if __name__ == "__main__":
    train()
