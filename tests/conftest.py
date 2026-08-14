"""pytest 共享配置：把 src 加入 sys.path，并提供数据集 fixture。"""

import os
import sys

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(scope="session")
def dataset():
    """加载已缓存的 StructureDataset（data/processed 下的图数据）。"""
    from datasets.structure_dataset import StructureDataset

    return StructureDataset(root=os.path.join(ROOT_DIR, "data"))
