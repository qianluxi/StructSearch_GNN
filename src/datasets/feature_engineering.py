import numpy as np


class FeatureEngineering:
    """
    负责：
    1. 坐标平移中心化
    2. PCA旋转对齐（旋转不变）
    3. 构建节点特征向量（保留原始尺度）
    """

    def __init__(self):
        pass

    # =========================================================
    # 主接口
    # =========================================================
    def build(self, nodes):
        """
        nodes: List[NodeDict]

        返回：
            node_features : (N, F)
            aligned_pos   : (N, 2)
        """

        if len(nodes) == 0:
            return None, None

        # -------------------------
        # 1. 原始质心坐标
        # -------------------------
        centroids = np.array(
            [n["centroid"] for n in nodes],
            dtype=float
        )

        # -------------------------
        # 2. 平移中心化
        # -------------------------
        centered = self._center(centroids)

        # -------------------------
        # 3. PCA旋转对齐
        # -------------------------
        aligned_pos = self._pca_align(centered)

        # -------------------------
        # 4. 构建节点特征
        # -------------------------
        node_features = self._build_node_features(nodes)

        return node_features, aligned_pos

    # =========================================================
    # 平移中心化
    # =========================================================
    def _center(self, pos):
        """
        将图平移到质心为原点
        """
        mean = pos.mean(axis=0, keepdims=True)
        return pos - mean

    # =========================================================
    # PCA旋转对齐
    # =========================================================
    def _pca_align(self, pos):
        """
        使用PCA主方向作为x轴，实现旋转不变
        """

        if len(pos) <= 1:
            return pos

        # 协方差矩阵
        cov = np.cov(pos.T)

        # 特征分解
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 取最大特征值对应特征向量
        idx = np.argmax(eigvals)
        principal_axis = eigvecs[:, idx]

        # 保证方向一致（消除符号翻转不确定性）
        if principal_axis[0] < 0:
            principal_axis = -principal_axis

        # 构造旋转矩阵
        x_axis = principal_axis
        y_axis = np.array([-x_axis[1], x_axis[0]])

        R = np.stack([x_axis, y_axis], axis=1)

        # 旋转
        aligned = pos @ R

        return aligned

    # =========================================================
    # 节点特征构建
    # =========================================================
    def _build_node_features(self, nodes):
        """
        构建节点特征向量（9维）

        Feature layout
        --------------------------
        0  column flag
        1  short wall flag
        2  shear wall flag

        3  area

        4  L_long
        5  L_short

        6  long_leg_length
        7  long_leg_thickness
        8  r (log1p 压缩)
        """

        features = []

        for n in nodes:

            # -------------------------
            # 类型 one-hot
            # -------------------------
            t = int(n["type"])

            one_hot = np.zeros(3, dtype=float)

            if 0 <= t <= 2:
                one_hot[t] = 1.0

            # -------------------------
            # 基本几何
            # -------------------------
            area = float(n["area"])

            Lx = float(n["Lx"])
            Ly = float(n["Ly"])

            L_long = max(Lx, Ly)
            L_short = min(Lx, Ly)

            # -------------------------
            # 墙肢参数
            # (来自 dxf_parser)
            # -------------------------
            long_leg_length = float(n["long_leg_length"])
            long_leg_thickness = float(n["long_leg_thickness"])
            r = float(n["r"])

            # r 动态范围极大（可到 1e8），直接标准化会被少数离群值主导，
            # 且 float32 下无法精确反标准化；用 log1p 压缩尾部。
            # 注意：仅压缩特征值，不影响 dxf_parser 中 classify_node 使用的物理 r。
            r_feat = float(np.log1p(max(r, 0.0)))

            geom_features = np.array([
                area,
                L_long,
                L_short,
                long_leg_length,
                long_leg_thickness,
                r_feat
            ], dtype=float)

            node_feature = np.concatenate([one_hot, geom_features])

            features.append(node_feature)

        return np.array(features, dtype=np.float32)


