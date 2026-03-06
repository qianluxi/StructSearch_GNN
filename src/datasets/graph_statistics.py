import numpy as np
from scipy.spatial.distance import pdist, squareform


class GraphStatistics:
    """
    图级统计特征构建（改进版）

    最终输出特征顺序（共 11 维）：

    0-2  : type_ratio (3)
    3-5  : type_area_ratio (3)
    6    : log_total_area (1)
    7    : area_cv (1)
    8    : area_entropy (1)
    9    : nn_mean (1)
    10   : nn_std (1)
    """

    def __init__(self, nodes, aligned_pos=None):
        """
        nodes: List[NodeDict]
        aligned_pos: (N,2) numpy array
                     已做平移中心化 + PCA对齐后的坐标
                     如果为 None，则使用原始 centroid
        """
        self.nodes = nodes
        self.N = len(nodes)

        if aligned_pos is not None:
            self.pos = aligned_pos
        else:
            # fallback（不推荐，但保证兼容）
            self.pos = np.array([n["centroid"] for n in nodes], dtype=float)

    # -------------------------------------------------
    # 1️⃣ 类型比例 P(G)
    # -------------------------------------------------
    def type_ratio(self):
        if self.N == 0:
            return np.zeros(3, dtype=np.float32)

        counts = np.zeros(3, dtype=float)

        for n in self.nodes:
            t = int(n["type"])
            if 0 <= t <= 2:
                counts[t] += 1

        return (counts / self.N).astype(np.float32)

    # -------------------------------------------------
    # 2️⃣ 面积比例 Q(G)
    # -------------------------------------------------
    def type_area_ratio(self):
        if self.N == 0:
            return np.zeros(3, dtype=np.float32)

        total_area = sum(float(n["area"]) for n in self.nodes)
        if total_area <= 1e-12:
            return np.zeros(3, dtype=np.float32)

        areas = np.zeros(3, dtype=float)
        for n in self.nodes:
            t = int(n["type"])
            if 0 <= t <= 2:
                areas[t] += float(n["area"])

        return (areas / total_area).astype(np.float32)

    # -------------------------------------------------
    # 3️⃣ 总面积（log1p）
    # -------------------------------------------------
    def log_total_area(self):
        if self.N == 0:
            return np.array([0.0], dtype=np.float32)

        total = sum(float(n["area"]) for n in self.nodes)
        log_total = np.log1p(total)  # log(1 + total_area)
        return np.array([log_total], dtype=np.float32)

    # -------------------------------------------------
    # 4️⃣ 面积分布特征：CV + 熵
    # -------------------------------------------------
    def area_distribution(self):
        if self.N == 0:
            return np.zeros(2, dtype=np.float32)

        areas = np.array([float(n["area"]) for n in self.nodes], dtype=float)
        if areas.size == 0:
            return np.zeros(2, dtype=np.float32)

        mean_area = areas.mean()
        std_area = areas.std()

        # 变异系数 CV（无量纲），防止方差量级过大
        cv = (std_area / mean_area) if mean_area > 1e-12 else 0.0

        ent = self._entropy(areas)

        return np.array([cv, ent], dtype=np.float32)

    def _entropy(self, x):
        s = x.sum()
        if s <= 1e-12:
            return 0.0

        p = x / s
        p = p[p > 0]

        return float(-np.sum(p * np.log(p)))

    # -------------------------------------------------
    # 5️⃣ 最近邻距离统计
    # -------------------------------------------------
    def nn_stats(self):
        if self.N <= 1:
            return np.zeros(2, dtype=np.float32)

        D = squareform(pdist(self.pos))
        np.fill_diagonal(D, np.inf)

        mins = D.min(axis=1)

        mean = mins.mean()
        std = mins.std()  # 替换原来的 var

        return np.array([mean, std], dtype=np.float32)

    # -------------------------------------------------
    # 6️⃣ 构建总图特征向量
    # -------------------------------------------------
    def build_feature_vector(self):
        features = []

        # 0-2 类型比例
        features.extend(self.type_ratio())

        # 3-5 面积比例
        features.extend(self.type_area_ratio())

        # 6 总面积的自然对数ln
        features.extend(self.log_total_area())

        # 7-8 面积 CV + 熵
        features.extend(self.area_distribution())

        # 9-10 最近邻均值 + std
        features.extend(self.nn_stats())

        return np.array(features, dtype=np.float32)
