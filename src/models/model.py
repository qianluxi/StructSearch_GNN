import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, node_dim=9, hidden_dim=64, emb_dim=128, num_classes=3):
        """
        node_dim: 节点特征维度
        hidden_dim: GNN 隐层维度
        emb_dim: 图嵌入维度
        num_classes: 分类类别数
        """
        super().__init__()

        # -----------------------
        # GNN layers (3-layer SAGEConv)
        # -----------------------
        self.conv1 = SAGEConv(node_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        # -----------------------
        # graph embedding
        # -----------------------
        self.lin = Linear(hidden_dim, emb_dim)

        # -----------------------
        # 分类头
        # -----------------------
        self.classifier = Linear(emb_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # GNN forward
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # graph pooling
        emb = global_mean_pool(x, batch)  # 图嵌入

        # final embedding
        emb = self.lin(emb)

        # 分类 logits
        logits = self.classifier(emb)

        return emb, logits
