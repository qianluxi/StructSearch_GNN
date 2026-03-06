# test_build_index.py
from src.retrieval.index_builder import build_index

def main():
    """
    测试构建索引
    """
    print("[Test] 开始测试索引构建...")
    build_index("configs/retrieval.yaml")
    print("[Test] 测试完成")

if __name__ == "__main__":
    main()
