import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from src.dataset import StructureDataset
from src.model import StructureGNN
import src.config as cfg

def build_index():
    print("正在加载数据集和模型...")
    dataset = StructureDataset(root=cfg.DATA_ROOT)
    
    # 这里不需要 shuffle，我们需要按顺序对应文件名
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = StructureGNN(
        num_node_features=dataset.num_node_features,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ).to(cfg.DEVICE)
    
    # 加载训练好的权重
    if not os.path.exists(cfg.MODEL_SAVE_PATH):
        print("错误: 未找到模型文件，请先运行 train.py")
        return

    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH))
    model.eval()
    
    database_embeddings = []
    database_filenames = []
    database_labels = []

    print("开始提取特征向量...")
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(cfg.DEVICE)
            
            # Forward pass: 获取 embedding
            _, embedding = model(data.x, data.edge_index, data.batch)
            
            # embedding 是 [1, 64] 的 tensor
            # 转为 numpy 数组保存
            vec = embedding.cpu().numpy()[0]
            
            # 获取文件名 (dataset中存储了)
            # DataLoader 会把 list of strings 变成 list of list，所以要取 [0]
            filename = data.filename[0]
            label = data.y.item()
            
            database_embeddings.append(vec)
            database_filenames.append(filename)
            database_labels.append(label)

            if (i+1) % 10 == 0:
                print(f"  已处理 {i+1}/{len(dataset)}")

    # 保存到文件
    save_dir = os.path.join(cfg.DATA_ROOT, 'feature_index')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    np.save(os.path.join(save_dir, 'embeddings.npy'), np.array(database_embeddings))
    np.save(os.path.join(save_dir, 'filenames.npy'), np.array(database_filenames))
    np.save(os.path.join(save_dir, 'labels.npy'), np.array(database_labels))
    
    print(f"建库完成！索引已保存至 {save_dir}")

if __name__ == '__main__':
    build_index()