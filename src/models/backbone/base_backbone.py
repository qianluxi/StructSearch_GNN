import torch.nn as nn

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        """
        输入: PyG Data
        输出: 图级 embedding (batch_size, hidden_dim)
        """
        raise NotImplementedError
