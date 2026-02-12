import argparse
from src.retrieval.index_builder import build_index
from src.retrieval.search_engine import SearchEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--search", type=str)

    args = parser.parse_args()

    if args.build:
        build_index()

    elif args.search:
        engine = SearchEngine()
        engine.search(args.search)

    else:
        print("请指定 --build 或 --search")


if __name__ == "__main__":
    main()
