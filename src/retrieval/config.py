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


def _resolve_config_paths(cfg):
    """把配置里的相对路径统一解析为相对项目根目录的绝对路径。

    Args:
        cfg: load_config 返回的 SimpleNamespace

    Returns:
        同一个 cfg 对象（相对路径字段被改写为绝对路径）。
    """
    relative_keys = (
        ("data_root",),
        ("checkpoint_dir",),
        ("embedding_dir",),
        ("data", "root"),
        ("data", "processed"),
        ("data", "splits"),
    )

    for keys in relative_keys:
        node = cfg
        for key in keys[:-1]:
            node = getattr(node, key, None)
            if node is None:
                break
        if node is None:
            continue

        value = getattr(node, keys[-1], None)
        if isinstance(value, str) and not os.path.isabs(value):
            setattr(node, keys[-1], os.path.join(ROOT_DIR, value))

    return cfg


def load_retrieval_config(cfg_path=None):
    """加载检索配置，返回 SimpleNamespace（支持 cfg.model.hidden_dim 式访问）。

    配置中的相对路径（data_root / checkpoint_dir / embedding_dir / data.*）
    会统一解析为相对项目根目录的绝对路径，调用方不再依赖当前工作目录。
    """
    cfg_path = cfg_path or DEFAULT_CONFIG_PATH
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(ROOT_DIR, cfg_path)
    return _resolve_config_paths(load_config(cfg_path))
