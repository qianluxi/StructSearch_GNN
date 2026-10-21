# SSIG: A Visually-Guided Graph Edit Distance for Floor Plan Similarity

**作者：** Casper C. J. van Engelenburg, Seyran Khademi, Jan C. van Gemert (Delft University of Technology)
**出处：** ICCV 2023 Workshop (CVAAD)
**原文：** `vanEngelenburg2023_SSIG_FloorPlan.pdf`
**状态：** 已下载原文（开放获取，CVF）
**技术路线：** 无学习结构相似度(SSIG) | statistics（支持统计/无学习路线）

## 中文摘要

本文提出一种简单而有效的、**无需学习（without the need for learning）** 的度量，用于衡量建筑平面图之间的**结构相似性**；定性结果显示其检索结果与深度学习方法相当。作者指出：比较平面图不能只看像素，关键在于布局各子部分（subdivisions）之间的**形状与关系**。当前主流做法使用**深度度量学习**，以 IoU 得到的相似标签训练成对向量空间，但 IoU 缺乏结构意识；为弥补这一点又引入**图匹配网络（GMN）**，然而 GMN 需要成对推理，在检索场景中不实用。为此，作者提出 **SSIG（Structural Similarity by IoU and GED）**，同时结合**图像距离（IoU）**与**图距离（图编辑距离 GED）**，并设计了一种高效排序算法，可在大规模平面图库（RPLAN）上按结构相似度排序。实验显示：对单个未见样本，本方法在单 CPU 上仅需约 **1–5 秒**完成排序，而 LayoutGMN 约需 **1500 秒**——计算效率远高于 GMN。

**关键词：** 结构相似性；平面图检索；图编辑距离；IoU；无学习度量

## 与本项目的关系

**与本项目核心结论高度呼应**：论证了"**无需深度学习的简单框架、只要目标与结构相似性对齐，即可媲美深度方法**"，为本项目"11 维统计描述子优于 GNN 表征"的结论提供旁证和方法论支持；其"GMN 成对推理不适合检索"的观察，也支持本项目不将图匹配/GNN 作为主线检索手段。
