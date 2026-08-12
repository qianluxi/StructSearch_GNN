"""检索配置加载。

基于 utils.config_loader.load_config 提供检索默认配置路径与类别映射，
避免在各检索模块中硬编码路径。
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.config_loader import load_config

# 默认检索配置路径（configs/retrieval.yaml）
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "retrieval.yaml")

# 标签 -> 中文名（与 StructureDataset 的 class_map 保持一致）
CLASS_NAMES = {
    0: "剪力墙",
    1: "框剪",
    2: "框架",
}

# 文件名关键字 -> 标签（与 StructureDataset 的 class_map 保持一致）
CLASS_MAP = {
    "剪力墙": 0,
    "框剪": 1,
    "框架": 2,
    "WALL": 0,
    "FRAME_SHEAR": 1,
    "FRAME": 2,
}


def load_retrieval_config(cfg_path=None):
    """加载检索配置，返回 SimpleNamespace（支持 cfg.model.hidden_dim 式访问）。"""

    cfg_path = cfg_path or DEFAULT_CONFIG_PATH
    return load_config(cfg_path)
