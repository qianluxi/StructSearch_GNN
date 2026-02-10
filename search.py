import os
import torch
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data, Batch
from sklearn.neighbors import kneighbors_graph

# 复用之前的代码
from src.dxf_utils import parse_dxf_file
from src.model import StructureGNN
import src.config as cfg

def process_single_dxf(filepath):
    """
    将单个DXF文件转化为模型可接受的 Data 对象
    (逻辑完全复制自 dataset.py 的 process 方法，但针对单文件)
    """
    # 1. 解析
    try:
        polygons, centroids, raw_features = parse_dxf_file(filepath)
    except Exception as e:
        print(f"解析错误: {e}")
        return None

    if len(polygons) < 2:
        print("构件过少")
        return None

    # 2. 归一化
    centroid_x = centroids[:, 0]
    centroid_y = centroids[:, 1]
    min_x, max_x = np.min(centroid_x), np.max(centroid_x)
    min_y, max_y = np.min(centroid_y), np.max(centroid_y)
    width = max_x - min_x
    height = max_y - min_y
    scale = max(width, height) if max(width, height) > 0 else 1.0
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    norm_pos_x = (centroid_x - center_x) / scale
    norm_pos_y = (centroid_y - center_y) / scale
    bbox_area = width * height if width * height > 0 else 1.0
    norm_area = raw_features[:, 0] / bbox_area 
    aspect_ratio = raw_features[:, 1]
    log_aspect_ratio = np.log(aspect_ratio + 1e-6)

    # 3. 特征
    x_features = np.stack([norm_area, log_aspect_ratio, norm_pos_x, norm_pos_y], axis=1)
    x = torch.tensor(x_features, dtype=torch.float)

    # 4. 边
    num_nodes = len(polygons)
    k = 5 if num_nodes > 5 else num_nodes - 1
    if k > 0:
        adj_matrix = kneighbors_graph(centroids, n_neighbors=k, mode='connectivity', include_self=False)
        edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 5. 封装 (不需要 y 和 filename)
    data = Data(x=x, edge_index=edge_index)
    return data

def search(target_dxf_path):
    # 1. 加载模型
    model = StructureGNN(4, cfg.HIDDEN_CHANNELS, cfg.NUM_CLASSES).to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH))
    model.eval()
    
    # 2. 加载数据库
    index_dir = os.path.join(cfg.DATA_ROOT, 'feature_index')
    try:
        db_embs = np.load(os.path.join(index_dir, 'embeddings.npy'))
        db_files = np.load(os.path.join(index_dir, 'filenames.npy'))
        db_labels = np.load(os.path.join(index_dir, 'labels.npy'))
    except FileNotFoundError:
        print("错误: 索引文件不存在，请先运行 build_index.py")
        return

    # 3. 处理输入文件
    print(f"正在分析文件: {target_dxf_path} ...")
    data = process_single_dxf(target_dxf_path)
    if data is None:
        return

    # 4. 模型推理
    data = data.to(cfg.DEVICE)
    # PyG 的模型通常期望 batch 维度，即使只有一个图
    # 构造一个 batch=[0, 0, ..., 0]
    batch = torch.zeros(data.x.size(0), dtype=torch.long).to(cfg.DEVICE)
    
    with torch.no_grad():
        logits, query_emb = model(data.x, data.edge_index, batch)
        
        # --- Task 1: 分类结果 ---
        pred_idx = logits.argmax(dim=1).item()
        # 反转 mapping
        idx_to_class = {v: k for k, v in cfg.CLASS_MAP.items()}
        pred_class_name = idx_to_class.get(pred_idx, "未知")
        
        prob = torch.softmax(logits, dim=1)[0][pred_idx].item()
        
        print("\n" + "="*40)
        print(f"【分类结果】: {pred_class_name} (置信度: {prob:.2%})")
        print("="*40)

        # --- Task 2: 相似度检索 ---
        # query_emb: [1, 64], db_embs: [N, 64]
        query_vec = query_emb.cpu().numpy()
        
        # 计算余弦相似度
        sims = cosine_similarity(query_vec, db_embs)[0] # 得到 [N] 数组
        
        # 排序 (从大到小)
        # argsort 返回的是从小到大的索引，所以要[::-1]
        sorted_indices = np.argsort(sims)[::-1]
        
        print(f"\n【最相似的历史案例 Top 5】:")
        for i in range(5):
            if i >= len(db_files): break
            idx = sorted_indices[i]
            
            # 过滤掉自己 (如果输入文件就在库里)
            if os.path.basename(target_dxf_path) == db_files[idx]:
                continue
                
            score = sims[idx]
            fname = db_files[idx]
            flabel = idx_to_class.get(db_labels[idx], "未知")
            
            print(f"{i+1}. {fname}")
            print(f"   - 相似度: {score:.4f}")
            print(f"   - 类型: {flabel}")
            print("-" * 20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input DXF file')
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        search(args.input)
    else:
        print(f"文件未找到: {args.input}")