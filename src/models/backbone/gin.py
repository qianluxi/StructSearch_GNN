import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_mean_pool

from .base_backbone import BaseBackbone


class GINBackbone(BaseBackbone):
    """
    GIN Backbone for graph embedding.

    Output:
        graph_embedding: [batch_size, hidden_dim]
    """

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.model.hidden_dim
        self.num_layers = cfg.model.num_layers
        self.num_node_features = getattr(cfg.model, "num_node_features", 4)

        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(self.num_layers):
            in_dim = (
                self.num_node_features
                if layer == 0
                else self.hidden_dim
            )

            mlp = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

            self.gnn_layers.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.pool = global_mean_pool

    def forward(self, data):
        """
        Args:
            data: PyG Batch object
                  data.x
                  data.edge_index
                  data.batch

        Returns:
            graph_embedding: Tensor [batch_size, hidden_dim]
        """

        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.gnn_layers, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        graph_embedding = self.pool(x, batch)

        return graph_embedding
