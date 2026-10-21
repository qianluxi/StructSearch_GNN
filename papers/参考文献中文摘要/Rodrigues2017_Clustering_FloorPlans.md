# Clustering of Architectural Floor Plans: A Comparison of Shape Representations

**作者：** Eugénio Rodrigues, David Sousa-Rodrigues, Mafalda Teixeira de Sampayo, Adélio Rodrigues Gaspar, Álvaro Gomes, Carlos Henggeler Antunes
**出处：** Automation in Construction, 2017, 80: 48–65
**DOI：** 10.1016/j.autcon.2017.03.017
**原文：** `Rodrigues2017_Clustering_FloorPlans.pdf`
**状态：** 已下载原文（开放获取，机构库版）
**技术路线：** 几何形状描述子 | statistics

## 中文摘要

生成式设计方法能产生大量建筑平面图方案，可能超出决策者的处理能力，因此需要能把生成数据"有意义地组织起来"的工具。本文在**无监督聚类**任务上，对四种建筑**形状表征**进行对比分析：其中三种基于已知描述子——**点距离（Point Distance）**、**转向函数（Turning Function）**、**基于网格的模型（Grid-Based）**；第四种是作者新提出的**切线距离（Tangent Distance）**，通过计算轮廓切线与形状几何中心的距离来表征（专为捕捉正交平面图形状而设计）。采用 Ward 链接的层次凝聚聚类，对 72 个合成平面图进行聚类并与参考划分比较。结果显示：**切线距离描述子表现最佳（Rand 指数 0.873）**；点距离与转向函数也有良好的感知效果；**基于网格的描述子最差**（对同一形状的比例变化过于敏感，导致被分散到多个簇）。

**关键词：** 无监督聚类；平面图设计；层次聚类；形状表征；描述子

## 与本项目的关系

直接比较了多种**几何形状描述子**，与本项目 11 维统计描述子中的"质心最近邻距离 / 面积分布 / 面积熵"属于同类思路；其结论"**描述子的选择显著决定聚类/检索质量**"可为本项目的特征设计与消融实验提供参照，其中"基于网格的描述子对比例敏感"的教训也提示：空间特征需谨慎设计（正是本项目 Spatial 组贡献有限的成因之一）。
