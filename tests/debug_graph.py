import os
import sys
import torch

# ---------------------------------------------------
# 确保可以找到 src
# ---------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from datasets.structure_dataset import StructureDataset

# ===================================================
# Graph Feature 打印
# ===================================================

def print_graph_features(data, dataset):

    print("\n===================================================")
    print("Graph-level Features")
    print("===================================================")

    norm_feat = data.graph_feature

    print("\n[Normalized Feature Vector] (用于模型训练)")
    print(norm_feat)
    print()

    print("mean =", norm_feat.mean().item())
    print("std  =", norm_feat.std().item())

    raw = dataset.inverse_transform(norm_feat)

    print("\n[Recovered Physical Feature]")
    print(raw)
    print()

    print("------ 物理意义 ------")

    print("节点类型比例:")
    print(f"  柱比例             = {raw[0]:.4f}")
    print(f"  短肢墙比例         = {raw[1]:.4f}")
    print(f"  剪力墙比例         = {raw[2]:.4f}")

    print()

    print("面积比例:")
    print(f"  柱面积比例         = {raw[3]:.4f}")
    print(f"  短肢墙面积比例     = {raw[4]:.4f}")
    print(f"  剪力墙面积比例     = {raw[5]:.4f}")

    print()

    print("整体规模:")
    print(f"  log(总竖向构件面积) = {raw[6]:.4f}")

    print()

    print("面积分布:")
    print(f"  面积变异系数 CV     = {raw[7]:.4f}")
    print(f"  面积熵 entropy      = {raw[8]:.4f}")

    print()

    print("空间分布:")
    print(f"  最近邻距离均值      = {raw[9]:.4f}")
    print(f"  最近邻距离标准差    = {raw[10]:.4f}")


# ===================================================
# Node Feature 打印
# ===================================================

def print_node_table(data):

    print("\n===================================================")
    print("Node Feature Table")
    print("===================================================")

    x = data.x
    pos = data.pos

    N = x.shape[0]

    print(f"节点数量: {N}")
    print()

    header = (
        "ID | Col SW Wall | "
        "Area | L_long | L_short | "
        "LongLeg | Thickness | r | "
        "PosX | PosY"
    )

    print(header)
    print("-" * len(header))

    for i in range(N):

        feat = x[i]
        px, py = pos[i]

        print(
            f"{i:02d} | "
            f"{int(feat[0])}   {int(feat[1])}   {int(feat[2])} | "
            f"{feat[3]:8.1f} | "
            f"{feat[4]:6.1f} | "
            f"{feat[5]:6.1f} | "
            f"{feat[6]:7.1f} | "
            f"{feat[7]:9.1f} | "
            f"{feat[8]:6.2f} | "
            f"{px:8.1f} | "
            f"{py:8.1f}"
        )


# ===================================================
# Edge 信息
# ===================================================

def print_edge_info(data):

    print("\n===================================================")
    print("Graph Edges")
    print("===================================================")

    edge_index = data.edge_index
    edge_attr = data.edge_attr

    E = edge_index.shape[1]

    print("边数量:", E)
    print()

    header = "EdgeID | Node_i -> Node_j | Distance"
    print(header)
    print("-" * len(header))

    for i in range(E):

        n1 = edge_index[0, i].item()
        n2 = edge_index[1, i].item()

        dist = edge_attr[i][0].item()

        print(f"{i:03d} | {n1:02d} -> {n2:02d} | {dist:10.2f}")


# ===================================================
# Graph Summary
# ===================================================

def print_graph_summary(data):

    print("\n===================================================")
    print("Graph Summary")
    print("===================================================")

    print("File:", data.filename)

    print()

    print("Nodes :", data.x.shape[0])
    print("Edges :", data.edge_index.shape[1])

    print()

    print("Node feature dim :", data.x.shape[1])
    print("Edge feature dim :", data.edge_attr.shape[1])
    print("Graph feature dim:", data.graph_feature.shape[0])


# ===================================================
# 主程序
# ===================================================

dataset = StructureDataset(root="data")

data = dataset[0]

print_graph_summary(data)

print_graph_features(data, dataset)

print_node_table(data)

print_edge_info(data)
