# Runlength Histogram Image Signature for Perceptual Retrieval of Architectural Floor Plans

**作者：** Lluís-Pere de las Heras, David Fernández, Alicia Fornés, Ernest Valveny, Gemma Sánchez, Josep Lladós
**出处：** Computer Vision Center, Universitat Autònoma de Barcelona；LNCS（GREC 会议论文），2014
**原文：** `delasHeras2014_RunlengthHistogram_FloorPlans.pdf`
**状态：** 已下载原文（开放获取）
**技术路线：** 感知统计描述子 | statistics

## 中文摘要

本文提出一种**行长度直方图（runlength histogram）签名**，作为建筑图纸的**感知描述子**用于检索。建筑图纸的"风格"可由对线条、形状、纹理的感知来刻画，而这些视觉刺激正是"空间属性、对称性、密度"等语义概念的基础。作者分别在**垂直、水平和对角方向**提取行长度直方图，用以刻画平面图中的线条与空间属性——可粗略对应"墙体"与"房间结构"的描述。在检索应用中，给定一张查询图纸，从库中返回相似图纸，并用基于**人工观察构建的 ground truth** 验证假设；此外还定性地给出在**建筑立面草图**上的检索结果。实验表明：该描述子虽简单，却能较好地适应两类不同的草图，为图形识别领域一条有挑战的研究方向。

**关键词：** 图形识别；图形检索；图像分类；行长度直方图

## 与本项目的关系

把"平面图的**线条/空间统计**"直接用作检索描述子，与本项目把结构构件几何**压缩为统计描述子**的思路一致；属于"**不识别具体构件、只做感知式统计**"的技术路线，可作为本项目"统计特征 vs 语义特征"的对照参考。其"用人工观察构建 ground truth 做评价"的做法，也与本项目构建专家评分基准的思路一致。
