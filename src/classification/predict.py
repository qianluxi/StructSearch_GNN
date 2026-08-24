"""结构类型识别入口：python -m src.classification.predict <dxf>"""

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from classification.classifier import StructureTypeClassifier
from utils.logging_setup import setup_logging


def main(argv=None):
    setup_logging()

    parser = argparse.ArgumentParser(
        description="结构类型识别（11D 统计特征，无 GNN）"
    )
    parser.add_argument("dxf", help="DXF 文件路径")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    result = StructureTypeClassifier(args.config).predict_dxf(args.dxf)
    if result is None:
        print("解析失败或未提取到构件节点")
        return 1

    print(f"结构类型: {result['type']} (type_id={result['type_id']})")
    print(
        f"置信度  : {result['confidence']:.2%} "
        f"(最近邻相似度 {result['score']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
