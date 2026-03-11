import os
import sys
import torch

# ---------------------------------------------------
# 确保可以找到 src
# ---------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from datasets.structure_dataset import StructureDataset

# ---------------------------------------------------
# 加载数据集
# ---------------------------------------------------
dataset = StructureDataset(root="data")

data = dataset[0]   # 选第1个 DXF 文件测试

print("===================================================")
print("File:", data.filename)
print("===================================================")

print("\n==== Graph-level Features (Normalized - For Training) ====")

norm_feat = data.graph_feature
print("Normalized vector:\n", norm_feat)
print()

print("Mean of normalized vector:", norm_feat.mean().item())
print("Std of normalized vector :", norm_feat.std().item())

# ---------------------------------------------------
# 反标准化（恢复物理量）
# ---------------------------------------------------
raw_recovered = dataset.inverse_transform(norm_feat)

print("\n==== Graph-level Features (Recovered Physical Values) ====")
print("Recovered vector:\n", raw_recovered)
print()

print("Type ratios:")
print("  柱个数比例:", raw_recovered[0].item())
print("  短肢墙个数比例:", raw_recovered[1].item())
print("  剪力墙个数比例:", raw_recovered[2].item())
print("  柱面积比例:", raw_recovered[3].item())
print("  短肢墙面积比例:", raw_recovered[4].item())
print("  剪力墙面积比例:", raw_recovered[5].item())
print("总竖向构件面积的自然对数:", raw_recovered[6].item())
print("面积变异系数 CV:", raw_recovered[7].item())
print("面积熵:", raw_recovered[8].item())
print("最近邻距离均值:", raw_recovered[9].item())
print("最近邻距离标准差:", raw_recovered[10].item())

print("\n==== Node-level Info ====")


def print_node_debug(nodes, node_features, pos):
    """
    打印节点详细信息
    """

    print("\n==== Node-level Info ====")
    print("节点数量:", len(nodes))
    print()

    for i, (node, feat) in enumerate(zip(nodes, node_features), 1):

        print(f"节点 {i}")

        print("  类型编码:")
        print(f"    柱 Column      = {int(feat[0])}")
        print(f"    短肢墙 ShortWall = {int(feat[1])}")
        print(f"    剪力墙 Wall     = {int(feat[2])}")

        print("  几何尺寸:")
        print(f"    面积 area = {feat[3]:.2f}")

        print("  尺寸:")
        print(f"    L_long  = {feat[4]:.2f}")
        print(f"    L_short = {feat[5]:.2f}")

        print("  墙肢参数:")
        print(f"    long_leg_length    = {feat[6]:.2f}")
        print(f"    long_leg_thickness = {feat[7]:.2f}")
        print(f"    r                  = {feat[8]:.2f}")

        x, y = pos[i-1]

        print("  位置坐标 (PCA对齐后):")
        print(f"    ({x:.2f}, {y:.2f})")

        print()

