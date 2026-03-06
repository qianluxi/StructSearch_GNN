# StructSearch GNN

**StructSearch** 是一个基于 **图神经网络 (Graph Neural Network, GNN)** 的结构体系识别与结构平面相似度分析系统。

系统将 **CAD DXF 结构平面图** 自动解析为 **结构构件图 (Structural Graph)**，并提取：

* **节点级特征 (Node Features)**
* **图级统计特征 (Graph-level Features)**

最终用于：

* 结构体系分类（剪力墙 / 框剪 / 框架）
* 结构平面相似度检索
* 结构布局特征分析

---

# 一、系统整体流程

整个数据处理与建模流程如下：

```
DXF 文件
   │
   ▼
DXF Parser
(dxf_parser.py)
   │
   ▼
Polygon 几何对象
   │
   ▼
节点特征工程
(feature_engineering.py)
   │
   ▼
图构建
(graph_builder.py)
   │
   ▼
图统计特征
(graph_statistics.py)
   │
   ▼
StructureDataset
(PyTorch Geometric Dataset)
   │
   ▼
Graph Neural Network
```

最终每个结构平面被表示为：

```
Graph G = (V, E)
```

其中：

```
V = 结构竖向构件节点
E = 构件之间的空间关系
```

---

# 二、节点定义 (Graph Nodes)

每个 **节点 (Node)** 表示一个 **竖向结构构件**：

```
柱
短肢剪力墙
剪力墙
```

节点来源：

```
DXF LWPOLYLINE 闭合多边形
```

每个多边形表示一个竖向构件截面。

---

# 三、节点特征向量 (Node Feature Vector)

每个节点使用 **9维特征向量** 表示：

```
x ∈ R⁹
```

## Node Feature Vector

| 维度 | 特征                 | 说明        |
| -- | ------------------ | --------- |
| 0  | type_column        | 是否为柱      |
| 1  | type_short_wall    | 是否为短肢剪力墙  |
| 2  | type_shear_wall    | 是否为剪力墙    |
| 3  | area               | 构件截面面积    |
| 4  | aspect_ratio       | 长宽比       |
| 5  | r                  | 长肢长度 / 厚度 |
| 6  | long_leg_length    | 长肢长度      |
| 7  | long_leg_thickness | 长肢厚度      |
| 8  | orientation        | 主方向编码     |

---

## 1 构件类型 (3维 One-Hot)

```
[type_column,
 type_short_wall,
 type_shear_wall]
```

例如：

柱

```
[1,0,0]
```

短肢墙

```
[0,1,0]
```

剪力墙

```
[0,0,1]
```

---

## 2 面积 (Area)

```
area = polygon_area
```

单位：

```
mm²
```

用于反映构件规模。

---

## 3 长宽比 (Aspect Ratio)

定义：

```
aspect_ratio = long_leg_length / short_leg_length
```

表示构件形状：

```
≈1  → 柱
>2  → 墙
```

---

## 4 r 值

定义：

```
r = long_leg_length / thickness
```

用于区分：

```
柱
短肢墙
剪力墙
```

典型范围：

```
柱        r ≈ 1
短肢墙    r ≈ 3~8
剪力墙    r > 8
```

计算过程：

1️⃣ 提取 polygon 边
2️⃣ 找到最长边方向
3️⃣ 合并平行边
4️⃣ 计算墙厚

---

## 5 长肢长度

```
long_leg_length
```

即构件长边长度。

---

## 6 长肢厚度

```
long_leg_thickness
```

即墙厚或柱宽。

---

## 7 方向特征

```
orientation
```

表示构件主方向：

```
0 → horizontal
1 → vertical
```

---

# 四、节点位置 (Node Position)

每个节点同时保存二维位置：

```
pos ∈ R²
```

```
(x, y)
```

表示：

```
构件几何中心
```

用于构建图连接关系。

---

# 五、图结构 (Graph Structure)

节点之间根据空间距离建立边。

当前支持两种连接方式：

### KNN连接

```
k = 6
```

每个节点连接最近的 `k` 个节点。

---

### Radius连接

```
radius = r
```

连接距离小于 `r` 的节点。

---

## 边特征

当前边特征为：

```
edge_attr = distance
```

表示：

```
节点中心距离
```

---

# 六、图级特征向量 (Graph-level Feature Vector)

每个结构图同时计算 **11维统计特征**：

```
g ∈ R¹¹
```

这些特征用于描述：

```
整体结构布局
```

---

## 1 构件数量比例 (3维)

```
柱数量比例
短肢墙数量比例
剪力墙数量比例
```

计算：

```
count(type) / total_nodes
```

---

## 2 构件面积比例 (3维)

```
柱面积比例
短肢墙面积比例
剪力墙面积比例
```

计算：

```
area(type) / total_area
```

---

## 3 总竖向构件面积 (log)

```
log(total_vertical_area)
```

用于描述：

```
结构整体规模
```

---

## 4 面积变异系数

```
CV = std(area) / mean(area)
```

表示：

```
构件尺寸均匀程度
```

---

## 5 面积熵

定义：

```
H = -Σ p_i log(p_i)
```

其中：

```
p_i = area_i / total_area
```

表示：

```
结构复杂度
```

---

## 6 最近邻距离均值

```
mean nearest neighbor distance
```

表示：

```
构件间距水平
```

---

## 7 最近邻距离标准差

```
std nearest neighbor distance
```

表示：

```
结构布局均匀性
```

---

# 七、图特征标准化

图级特征采用 **Dataset级标准化**：

```
x_norm = (x - mean) / std
```

训练时使用：

```
normalized feature
```

同时保存：

```
raw feature
```

可用于恢复物理量。

---

# 八、Dataset 输出

每个样本最终生成一个 **PyTorch Geometric Data 对象**：

```
Data(
    x               # node features
    edge_index      # graph edges
    edge_attr       # edge features
    pos             # node positions
    graph_feature   # normalized graph feature
    graph_feature_raw
    y               # label
)
```

---

# 九、项目代码结构

```
src/
 └── datasets
     ├── dxf_parser.py
     │
     ├── feature_engineering.py
     │
     ├── graph_builder.py
     │
     ├── graph_statistics.py
     │
     ├── structure_dataset.py
     │
     └── schemas.py
```

---

## dxf_parser.py

负责：

```
DXF → Polygon
Polygon → Node 属性
```

包括：

```
面积
r
长宽
类型
```

---

## feature_engineering.py

构建：

```
node feature vector
```

---

## graph_builder.py

构建：

```
edge_index
edge_attr
```

支持：

```
knn
radius
```

---

## graph_statistics.py

计算：

```
graph-level feature vector
```

---

## structure_dataset.py

实现：

```
PyTorch Geometric Dataset
```

功能：

```
DXF → Graph
Graph Feature Normalization
Dataset 构建
```

---

# 十、应用场景

该系统可用于：

### 结构体系识别

```
剪力墙
框剪
框架
```

---

### 结构平面相似度搜索

例如：

```
输入一个新平面
→ 找数据库中最相似结构
```

---

### 结构布局统计分析

例如：

```
构件密度
结构复杂度
构件尺寸分布
```

---

那会非常像一个 **科研项目 README**。
