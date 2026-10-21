# Automatic Analysis and Sketch-Based Retrieval of Architectural Floor Plans

**作者：** Sheraz Ahmed, Markus Weber, Marcus Liwicki, Christoph Langenhan, Andreas Dengel, Frank Petzold
**出处：** Pattern Recognition Letters, 2013, 35: 91–100
**DOI：** 10.1016/j.patrec.2013.04.005
**原文：** `Ahmed2013_Analysis_SketchRetrieval_FloorPlans.pdf`（10 页）
**状态：** ✅ 已下载原文
**技术路线：** 结构/语义特征检索 | statistics

## 中文摘要

建筑设计中，建筑师常以已建项目为参考：遇到新的建筑问题，其首要任务就是检索相似项目，通过研究一个或多个历史参考项目来推导当前方案。然而目前建筑领域的电子检索依赖**文本标注**，而纯文字描述主观、不精确，对平面图检索而言过于模糊。

本文贡献有二：**其一**，提出基于草图的检索系统 **a.SCatch**，用户只需在触控板（DFKI Touch & Write）上画出新平面图的一部分，系统即提取语义结构并进行检索；**其二**，提出一套完整的**平面图自动分析系统**，从图像中自动提取结构与语义内容，包括粗/中/细线分割、基于**凸包**的建筑边界提取（显著提升性能）、基于凸凹假设的墙边检测、以及基于 **SURF** 的门位置检测，并将内容表示为**图结构语义表示**。最后用图匹配比较并返回最相似的平面图。在参考数据集上的实验表明，该分析系统优于以往方法，草图识别与检索性能已足以实用。

**关键词：** 平面图分析；结构分析；墙体检测；符号识别；房间检测；图匹配

## 原文摘要（English）

> During the design process, architects use existing, already-designed buildings as reference. ... This article's contribution is twofold. First, a sketch-based system (**a.SCatch**) is proposed for querying a floor-plan repository. Second, a complete floor-plan analysis system is presented that automatically extracts structural and semantic content (thick/medium/thin line differentiation, building-boundary extraction via convex hull, wall-edge detection, SURF-based door detection), representing content as a **graph-based semantic structure**. Extracted structures are compared using graph matching, and the most similar plan is retrieved. Experiments on a reference data set show the analysis system outperforms previous systems, and retrieval performance is good enough for practical use.

## 与本项目的关系

**方法论高度同构**：自动解析图纸 → 提取结构/语义信息 → 图/特征表征 → 相似检索，与本项目"DXF 解析 → 特征工程 → 统计描述子 → 检索"管线一致。其"文本标注不足以表达结构相似性"的论点，与本项目用工程语义统计特征替代文本元数据检索的动机相同。
