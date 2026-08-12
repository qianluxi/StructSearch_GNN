# heads/classifier_head.py

import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, embedding):
        return self.fc(embedding)
