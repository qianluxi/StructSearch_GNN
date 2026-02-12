import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_mean_pool

class StructureGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(StructureGNN, self).__init__()
        
        # --- 第一层 GIN ---
        # GIN 的核心是一个 MLP，用来聚合邻居特征
        self.conv1 = GINConv(
            Sequential(
                Linear(num_node_features, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
        )
        
        # --- 第二层 GIN ---
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
        )
        
        # --- 第三层 GIN ---
        self.conv3 = GINConv(
            Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
        )

        # --- 分类头 (Classifier Head) ---
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # x: Node features [N, 4]
        # edge_index: Graph connectivity [2, E]
        # batch: Batch vector [N] (指示每个节点属于哪个图)

        # 1. 节点嵌入 (Node Embeddings)
        x = self.conv1(x, edge_index)
        x = x.relu()
        
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        x = self.conv3(x, edge_index)
        # 这里 x 的形状还是 [N, hidden_channels]

        # 2. 图嵌入 (Graph Embedding) - 关键步骤！
        # 将属于同一个图的所有节点特征取平均值
        # 结果 shape: [Batch_Size, hidden_channels]
        graph_embedding = global_mean_pool(x, batch)

        # 3. 分类 (Classification)
        x = F.dropout(graph_embedding, p=0.5, training=self.training)
        logits = self.lin(x)
        
        # 返回两个值：分类结果 logits 和 特征向量 embedding
        # 训练时用 logits 计算 Loss
        # 检索时用 embedding 计算相似度
        return logits, graph_embedding