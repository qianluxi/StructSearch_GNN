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
