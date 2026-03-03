import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.proj(x)
        x = F.normalize(x, dim=1)  # 用于 cosine similarity
        return x
