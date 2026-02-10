# GNN-Based Structural System Classification & Retrieval
基于图神经网络的建筑结构类型分类与图纸检索系统

## 📖 项目简介

本项目利用 **图神经网络 (Graph Neural Networks, GNN)** 技术，实现了对建筑结构施工图（DXF文件）的自动分类与相似案例检索。

传统方法往往依赖硬性的几何规则（如长宽比、面积阈值），难以处理复杂的拓扑结构和“模糊”的工程概念。本项目将建筑平面图转化为 **图结构数据 (Graph Data)**，利用 GNN 强大的特征提取能力，学习构件之间的空间关系与拓扑模式。

### 核心功能
1.  **结构类型分类 (Classification)**：
    *   自动识别图纸属于 **框架结构 (Frame)**、**框剪结构 (Frame-Shear)** 还是 **剪力墙结构 (Shear Wall)**。
    *   目前在测试集上准确率可达 **90% - 100%**。
2.  **相似案例检索 (Retrieval)**：
    *   基于训练好的模型提取图纸的 **高维特征向量 (Embedding)**。
    *   支持“以图搜图”：输入一张新图纸，快速检索历史数据库中最相似的工程案例。

## 🏗️ 技术方案

### 1. 数据预处理 (DXF to Graph)
*   **节点 (Nodes)**：提取 DXF 中的闭合多段线（代表柱、墙、异形柱等竖向构件）。
*   **节点特征 (Node Features)**：
    *   归一化面积 (Normalized Area)
    *   长宽比 (Aspect Ratio, Log-scale)
    *   相对位置 (Normalized Coordinates X, Y)
*   **边 (Edges)**：构建 KNN 图 (K=5)，连接空间上最近的 5 个邻居构件，捕获局部拓扑结构。

### 2. 模型架构
*   **Backbone**: GIN (Graph Isomorphism Network) / GCN
*   **Pooling**: Global Mean Pooling (获取全图特征向量)
*   **Head**: MLP Classifier (分类头)
*   **Loss**: Cross Entropy Loss (分类任务)

## 🚀 快速开始

### 环境依赖
```bash
pip install torch torch-geometric ezdxf shapely scikit-learn matplotlib numpy
```

### 数据准备
1.  在根目录下创建 `data/raw_dxf/` 文件夹。
2.  将 DXF 文件放入该文件夹，文件名需包含类别关键词（如 `项目A_框剪_01.dxf`）。
    *   支持类别：`剪力墙`, `框剪`, `框架`
3.  运行训练脚本，程序会自动解析 DXF 并生成图数据。

### 运行步骤

#### 1. 训练模型
```bash
python train.py
```
*   训练完成后，最佳模型会保存至 `checkpoints/best_model.pth`。
*   同时生成测试集准确率报告。

#### 2. 构建特征索引库
```bash
python build_index.py
```
*   提取所有历史图纸的特征向量，保存至 `data/feature_index/`。

#### 3. 检索相似图纸
```bash
python search.py --input data/raw_dxf/某个测试文件.dxf
```
*   输出预测的结构类型。
*   输出最相似的 Top 5 历史案例及其相似度。

#### 4. 特征可视化 (新增)
```bash
python visualize_features.py
```
*   生成 t-SNE / PCA 降维可视化图，展示不同结构类型在特征空间中的分布情况。

## 📂 项目结构

```text
StructSearch_GNN/
├── data/
│   ├── raw_dxf/           # [Input] 原始 DXF 文件
│   ├── processed/         # [Cache] PyG 处理后的图数据
│   └── feature_index/     # [Output] 检索用的向量库
├── src/
│   ├── dataset.py         # 数据加载与图构建
│   ├── dxf_utils.py       # DXF 几何解析工具
│   ├── model.py           # GNN 模型定义
│   └── config.py          # 全局配置参数
├── train.py               # 训练脚本
├── build_index.py         # 建库脚本
├── search.py              # 检索脚本
├── visualize_features.py  # 可视化分析工具
└── README.md              # 说明文档
```

## 📊 效果展示

*(此处可插入 visualize_features.py 生成的图片)*

*   **分类性能**：模型能有效区分长条形剪力墙密布的结构（剪力墙）和点状柱分布的结构（框架）。
*   **检索效果**：对于同类型的图纸，特征向量在空间中聚集紧密；不同类型的图纸被明显分开。

