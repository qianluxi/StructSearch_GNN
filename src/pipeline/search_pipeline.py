"""两阶段检索流水线编排。

Stage 1：StructureTypeClassifier（11D 统计特征）判断结构类型；
Stage 2：SearchEngine.search_by_type 在类型内做三路线相似检索。
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from classification.classifier import StructureTypeClassifier
from retrieval.search_engine import SearchEngine


class SearchPipeline:
    """将结构类型识别与类内相似检索串起来的两阶段入口。"""

    def __init__(self, cfg_path=None):
        self.cfg_path = cfg_path
        self.classifier = StructureTypeClassifier(cfg_path)
        self.engine = SearchEngine(cfg_path)

    def search(self, dxf_path, topk=None, method="hybrid"):
        """对单张 DXF 执行完整的两阶段检索。

        Args:
            dxf_path: DXF 文件路径
            topk: 类内返回的相似案例数
            method: "statistics" / "gnn" / "hybrid"

        Returns:
            dict：type / type_id / confidence / method / results
            （解析失败时返回 None）
        """
        topk = topk or getattr(self.engine.cfg.retrieval, "topk", 5)

        pred = self.classifier.predict_dxf(dxf_path)
        if pred is None:
            return None

        result = self.engine.search_by_type(
            pred["data"],
            type_id=pred["type_id"],
            topk=topk,
            method=method,
        )
        result["type"] = pred["type"]
        result["type_id"] = pred["type_id"]
        result["confidence"] = pred["confidence"]
        result["data"] = pred["data"]
        return result
