import torch.nn as nn

from src.models.backbone.gin import GINBackbone
from src.models.heads.classifier_head import ClassifierHead
from src.models.heads.projection_head import ProjectionHead


class StructModel(nn.Module):
    """
    Unified Graph Model for:
    - Classification
    - Embedding extraction (retrieval)
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # =============================
        # 1. Backbone
        # =============================
        backbone_name = cfg.model.backbone.lower()

        if backbone_name == "gin":
            self.backbone = GINBackbone(cfg)
        else:
            raise NotImplementedError(
                f"Backbone {backbone_name} not implemented."
            )

        # =============================
        # 2. Heads
        # =============================

        # 分类头
        self.classifier_head = ClassifierHead(
            in_dim=cfg.model.hidden_dim,
            num_classes=cfg.model.num_classes
        )

        # 投影头（用于检索 embedding）
        self.projection_head = ProjectionHead(
            in_dim=cfg.model.hidden_dim,
            out_dim=cfg.embedding.dim
        )

    # ==================================================
    # 分类任务 forward
    # ==================================================
    def forward(self, data):
        """
        Returns:
            logits: [batch_size, num_classes]
        """
        graph_embedding = self.backbone(data)
        logits = self.classifier_head(graph_embedding)
        return logits

    # ==================================================
    # 检索任务 encode
    # ==================================================
    def encode(self, data):
        """
        Returns:
            logits: 分类输出
            projected_embedding: 用于检索的 embedding
        """
        graph_embedding = self.backbone(data)

        logits = self.classifier_head(graph_embedding)
        projected_embedding = self.projection_head(graph_embedding)

        return logits, projected_embedding
