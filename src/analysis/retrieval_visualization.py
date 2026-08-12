# src/analysis/retrieval_visualization.py

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

import networkx as nx

# ===================== 路径 =====================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder


# =========================================================
# 结构绘图
# =========================================================
def draw_structure(data, ax=None, title=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))

    x_raw = data.x_raw.cpu().numpy() if hasattr(data, "x_raw") else data.x.cpu().numpy()

    # 节点类型 one-hot 布局：0=Column(柱) 1=ShortWall(短肢墙) 2=Wall(剪力墙)
    node_type = np.argmax(x_raw[:, :3], axis=1)

    column_mask = node_type == 0
    short_wall_mask = node_type == 1
    wall_mask = node_type == 2

    # ========= 使用真实坐标 =========
    pos = data.pos.cpu().numpy()

    edge_index = data.edge_index.cpu().numpy()

    # edges
    for i, j in edge_index.T:
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color="gray",
            linewidth=0.7
        )

    # nodes
    if column_mask.any():
        ax.scatter(
            pos[column_mask, 0],
            pos[column_mask, 1],
            c="red",
            s=60,
            label="Column"
        )

    if short_wall_mask.any():
        ax.scatter(
            pos[short_wall_mask, 0],
            pos[short_wall_mask, 1],
            c="blue",
            s=60,
            label="ShortWall"
        )

    if wall_mask.any():
        ax.scatter(
            pos[wall_mask, 0],
            pos[wall_mask, 1],
            c="green",
            s=80,
            label="Wall"
        )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=9)

    return ax



# =========================================================
# cosine similarity
# =========================================================
def cosine_similarity_matrix(emb):

    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = emb @ emb.T

    return sim


# =========================================================
# 提取 embedding
# =========================================================
def extract_embeddings(dataset, model, device, batch_size=8):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    labels = []
    datas = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            emb, logits = model(batch.x, batch.edge_index, batch.batch)

            embeddings.append(emb.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

            for i in range(batch.num_graphs):
                datas.append(batch[i])

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    return embeddings, labels, datas


# =========================================================
# retrieval visualization
# =========================================================
def visualize_retrieval(
        top_k=5,
        num_queries=10,
        batch_size=8):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- dataset ----------------
    dataset = StructureDataset(root=os.path.join(ROOT_DIR, "data"))

    # ---------------- model ----------------
    checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "model.pt")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = GraphEncoder(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # ---------------- embeddings ----------------
    embeddings, labels, datas = extract_embeddings(
        dataset,
        model,
        device,
        batch_size,
    )

    print("Embedding shape:", embeddings.shape)

    # ---------------- similarity ----------------
    sim_matrix = cosine_similarity_matrix(embeddings)

    save_dir = os.path.join(ROOT_DIR, "fig", "retrieval_cases")
    os.makedirs(save_dir, exist_ok=True)

    # 只随机选部分 query，避免图太多
    query_indices = np.random.choice(
        len(dataset),
        size=min(num_queries, len(dataset)),
        replace=False,
    )

    for q_idx in query_indices:

        sim = sim_matrix[q_idx]

        # 排序
        topk_idx = np.argsort(-sim)[1: top_k + 1]

        print("\nQuery:", q_idx)
        print("Label:", labels[q_idx])
        print("\nTopK results:")

        for i, idx in enumerate(topk_idx):

            print(
                f"{i+1} | id={idx} label={labels[idx]} sim={sim[idx]:.4f}"
            )

        # -------- plot --------
        fig, axes = plt.subplots(
            1,
            top_k + 1,
            figsize=(3 * (top_k + 1), 3),
        )

        # query
        draw_structure(
            datas[q_idx],
            ax=axes[0],
            title=f"Query\nlabel={labels[q_idx]}",
        )

        # top-k
        for k, idx in enumerate(topk_idx):

            title = f"Top{k+1}\nlabel={labels[idx]}\nsim={sim[idx]:.2f}"

            draw_structure(
                datas[idx],
                ax=axes[k + 1],
                title=title,
            )

        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"retrieval_query_{q_idx}.png",
        )

        plt.savefig(save_path, dpi=200)
        plt.close()

    print("Saved to:", save_path)


def save_single_query_figure(
        dataset,
        query_dxf_path,
        result,
        save_dir,
        k=6,
        connect_mode="knn",
        query_data=None):
    """保存"查询图 + Top-K 相似案例"的对比图。

    Args:
        dataset: StructureDataset（检索库）
        query_dxf_path: 查询 DXF 文件路径
        result: SearchEngine.search() 返回的 dict（含 pred_class / results）
        save_dir: 输出目录
        k / connect_mode: 构图参数，须与建库时一致
        query_data: 查询图 Data 对象（search() 已解析好，避免重复解析）；
            为 None 时才回退到重新解析 DXF。

    Returns:
        save_path: 保存的图片路径；查询解析失败时返回 None。
    """
    from retrieval.processor import process_single_dxf

    # Windows 中文字体，避免图中中文显示为方框
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs(save_dir, exist_ok=True)

    if query_data is None:
        query_data = process_single_dxf(
            query_dxf_path,
            k=k,
            connect_mode=connect_mode
        )
    if query_data is None:
        print("[Figure] 查询图解析失败，跳过检索对比图")
        return None

    # 在库中按文件名找 Top-K 对应的 Data
    topk_datas = []
    for item in result.get("results", []):
        found = next(
            (
                d for d in dataset
                if getattr(d, "filename", None) == item["filename"]
            ),
            None,
        )
        topk_datas.append((item, found))

    topk = len(topk_datas)
    fig, axes = plt.subplots(
        1,
        topk + 1,
        figsize=(3 * (topk + 1), 3),
    )

    draw_structure(query_data, ax=axes[0], title="Query")

    for rank, (item, data) in enumerate(topk_datas, start=1):
        title = f"Top{rank}\nsim={item['score']:.2f}"
        if data is not None:
            draw_structure(data, ax=axes[rank], title=title)
        else:
            axes[rank].axis("off")
            axes[rank].set_title(title, fontsize=9)

    fig.suptitle(
        f"Query: {os.path.basename(query_dxf_path)}  |  "
        f"Pred: {result.get('pred_class', '')}",
        fontsize=10,
    )
    plt.tight_layout()

    stem = os.path.splitext(os.path.basename(query_dxf_path))[0]
    save_path = os.path.join(save_dir, f"query_{stem}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Figure] 检索结果图已保存: {save_path}")
    return save_path


# =========================================================
# main
# =========================================================
if __name__ == "__main__":

    visualize_retrieval(
        top_k=5,
        num_queries=10,
    )
