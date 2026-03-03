import os
import sys
import torch

# ---------------------------------------------------
# 确保可以找到 src
# ---------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from datasets.structure_dataset import StructureDataset


def main():

    dataset_root = os.path.join(ROOT_DIR, "data")

    print("=" * 60)
    print("Testing StructureDataset Pipeline")
    print("=" * 60)

    # ---------------------------------------------------
    # 构建数据集
    # ---------------------------------------------------
    dataset = StructureDataset(
        root=dataset_root,
        k=6,
        connect_mode="knn"
    )

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("❌ No valid graphs generated.")
        return

    # ---------------------------------------------------
    # 取一个样本测试
    # ---------------------------------------------------
    data = dataset[0]

    print("\n--- Sample Graph Info ---")
    print("\n--- All Files ---")
    for i in range(len(dataset)):
        print(i, dataset[i].filename)
    print(f"Label: {data.y.item()}")

    print("\nNode feature shape:", data.x.shape)
    print("Edge index shape:", data.edge_index.shape)

    if data.edge_attr is not None:
        print("Edge attr shape:", data.edge_attr.shape)

    print("Graph feature shape:", data.graph_feature.shape)
    print("Position shape:", data.pos.shape)

    # ---------------------------------------------------
    # 一致性检查
    # ---------------------------------------------------
    print("\n--- Consistency Checks ---")

    N = data.x.shape[0]

    assert data.pos.shape[0] == N, "pos 与 x 节点数不一致"
    assert data.graph_feature.shape[0] == 11, "图级特征维度应为 11"

    print("✔ Node count consistent")
    print("✔ Graph feature dimension correct (11)")
    print("✔ Pipeline OK")

    print("\nTest Completed Successfully.")


if __name__ == "__main__":
    main()
