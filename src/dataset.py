import os
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from sklearn.neighbors import kneighbors_graph
from .dxf_utils import parse_dxf_file

class StructureDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(StructureDataset, self).__init__(root, transform, pre_transform)
        # 加载处理好的数据
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()

    @property
    def raw_dir(self):
        # [关键修改] 告诉 PyG 原始目录是 raw_dxf 而不是默认的 raw
        return os.path.join(self.root, 'raw_dxf')

    @property
    def raw_file_names(self):
        # 扫描 raw_dxf 文件夹下的 dxf 文件
        if not os.path.exists(self.raw_dir):
            return []
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.dxf')]

    @property
    def processed_file_names(self):
        return ['structure_graph_data.pt']

    def download(self):
        pass

    def process(self):
        print("开始处理 DXF 文件并构建图数据...")
        
        data_list = []
        
        # 标签映射表：文件名包含关键词 -> 类别ID
        # 可以根据你的实际文件名修改
        class_map = {
            '剪力墙': 0, 
            '框剪': 1, 
            '框架': 2,
            # 可以添加更多别名
            'WALL': 0,
            'FRAME_SHEAR': 1,
            'FRAME': 2
        }

        files = self.raw_file_names
        if len(files) == 0:
            print("警告: raw_dxf 文件夹下没有找到 .dxf 文件！")
            return

        for filename in files:
            file_path = os.path.join(self.root, 'raw_dxf', filename)
            
            # 1. 确定标签 (Y)
            label = -1
            for key, val in class_map.items():
                if key in filename: # 只要文件名包含关键词
                    label = val
                    break
            
            # 如果文件名里没有关键词，你可以选择跳过，或者标记为未知
            if label == -1:
                print(f"跳过未分类文件: {filename}")
                continue

            # 2. 解析几何数据
            # polygons: 形状对象, centroids: 质心坐标(Nx2), raw_features: [面积, 长宽比](Nx2)
            polygons, centroids, raw_features = parse_dxf_file(file_path)
            
            if len(polygons) < 2:
                print(f"跳过空文件或构件过少: {filename}")
                continue

            # 3. 数据归一化 (非常重要！)
            # 我们需要把整个建筑的坐标缩放到 [-1, 1] 之间，保留相对位置
            # 同时把面积也归一化
            
            # 坐标归一化
            centroid_x = centroids[:, 0]
            centroid_y = centroids[:, 1]
            min_x, max_x = np.min(centroid_x), np.max(centroid_x)
            min_y, max_y = np.min(centroid_y), np.max(centroid_y)
            
            # 计算包围盒的宽高
            width = max_x - min_x
            height = max_y - min_y
            scale = max(width, height) if max(width, height) > 0 else 1.0
            
            # 几何中心
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            # 归一化后的坐标 (Nx2)
            norm_pos_x = (centroid_x - center_x) / scale
            norm_pos_y = (centroid_y - center_y) / scale
            
            # 面积归一化 (相对于整个建筑包围盒的面积)
            # 这样大楼里的大柱子和小楼里的小柱子会有区分
            bbox_area = width * height if width * height > 0 else 1.0
            norm_area = raw_features[:, 0] / bbox_area 
            
            # 长宽比不需要归一化，通常在 1.0 ~ 10.0 之间，取个对数让它分布更平滑
            aspect_ratio = raw_features[:, 1]
            log_aspect_ratio = np.log(aspect_ratio + 1e-6)

            # 4. 构建最终节点特征向量 X
            # 特征维度 = 4: [归一化面积, Log长宽比, 归一化X, 归一化Y]
            # 这些特征足够区分墙(长宽比大)、柱(长宽比小)、角柱(位置偏)、核心筒(面积大且在中间)
            x_features = np.stack([norm_area, log_aspect_ratio, norm_pos_x, norm_pos_y], axis=1)
            x = torch.tensor(x_features, dtype=torch.float)

            # 5. 构建边 (Edges) - KNN Graph
            # 每个构件连接最近的 K 个构件
            num_nodes = len(polygons)
            k = 5 if num_nodes > 5 else num_nodes - 1
            
            if k > 0:
                # mode='connectivity' 返回 0/1 矩阵
                adj_matrix = kneighbors_graph(centroids, n_neighbors=k, mode='connectivity', include_self=False)
                # 转换为 PyG 需要的 COO 格式 (2, Num_Edges)
                edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # 6. 封装 Data 对象
            data = Data(
                x=x, 
                edge_index=edge_index, 
                y=torch.tensor([label], dtype=torch.long),
                pos=torch.tensor(np.stack([norm_pos_x, norm_pos_y], axis=1), dtype=torch.float), # 保存归一化位置用于可视化
                filename=filename # 保存文件名方便检索时调用
            )
            
            data_list.append(data)

        if len(data_list) == 0:
            print("错误: 未能成功处理任何文件。")
            return

        # 7. 保存
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"成功处理 {len(data_list)} 个文件，保存至 {self.processed_paths[0]}")