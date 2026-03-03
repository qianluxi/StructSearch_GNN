import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.models.struct_model import StructModel
from src.retrieval.embedder import Embedder
from .processor import process_single_dxf
from .index_loader import load_feature_index
import src.config as cfg
from src.utils.config_loader import load_config


class SearchEngine:

    def __init__(self):

        # 1️⃣ 加载模型
        cfg = load_config("configs/retrieval.yaml")
        self.model = StructModel(cfg)

        self.model.load_state_dict(
            torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE)
        )

        # 2️⃣ Embedder
        self.embedder = Embedder(self.model, device=cfg.DEVICE)

        # 3️⃣ 加载数据库
        self.db_embs, self.db_files, self.db_labels = load_feature_index()

        self.idx_to_class = {v: k for k, v in cfg.CLASS_MAP.items()}

    def search(self, target_dxf_path):

        print(f"\n正在分析文件: {target_dxf_path}")

        # 1️⃣ DXF → graph
        data = process_single_dxf(target_dxf_path)
        if data is None:
            return

        # ---------- 分类 ----------
        data = data.to(cfg.DEVICE)

        with torch.no_grad():
            logits = self.model(data)

        pred_idx = logits.argmax(dim=1).item()
        pred_class_name = self.idx_to_class.get(pred_idx, "未知")
        prob = torch.softmax(logits, dim=1)[0][pred_idx].item()

        print("\n" + "=" * 40)
        print(f"【分类结果】: {pred_class_name} (置信度: {prob:.2%})")
        print("=" * 40)

        # ---------- 检索 ----------
        query_vec = self.embedder.extract_one(data)

        sims = cosine_similarity(
            query_vec.reshape(1, -1),
            self.db_embs
        )[0]

        sorted_indices = np.argsort(sims)[::-1]

        print("\n【最相似的历史案例 Top 5】")

        rank = 1
        for idx in sorted_indices:

            if rank > 5:
                break

            if os.path.basename(target_dxf_path) == self.db_files[idx]:
                continue

            score = sims[idx]
            fname = self.db_files[idx]
            flabel = self.idx_to_class.get(
                self.db_labels[idx],
                "未知"
            )

            print(f"{rank}. {fname}")
            print(f"   - 相似度: {score:.4f}")
            print(f"   - 类型: {flabel}")
            print("-" * 30)

            rank += 1
