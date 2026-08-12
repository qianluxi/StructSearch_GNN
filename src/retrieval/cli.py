"""检索命令行入口：建库 / 检索。"""

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from retrieval.index_builder import build_index
from retrieval.search_engine import SearchEngine


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="StructSearch_v2 检索工具：构建索引或检索单张 DXF"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="构建特征索引（embeddings.npy 等）"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="对指定 DXF 文件进行分类 + Top-K 检索"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--topk", type=int, default=None)

    args = parser.parse_args(argv)

    if args.build:
        build_index(args.config)
    elif args.search:
        engine = SearchEngine(args.config)
        engine.search(args.search, topk=args.topk)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
