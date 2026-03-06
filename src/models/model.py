import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphEncoder(nn.Module):

    def __init__(self, in_dim=7, hidden_dim=64, emb_dim=128):

        super().__init__()

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        self.lin = nn.Linear(hidden_dim, emb_dim)


    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x
