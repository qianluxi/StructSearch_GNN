# Semantic Pattern-based Retrieval of Architectural Floor Plans with Case-based and Graph-based Searching Techniques and their Evaluation and Visualization

**作者：** Qamer Uddin Sabri, Johannes Bayer, Viktor Ayzenshtadt, Syed Saqib Bukhari, **Klaus-Dieter Althoff**, Andreas Dengel
**出处：** ICPRAM 2017 (6th Int. Conf. on Pattern Recognition Applications and Methods), pp. 50–60, SCITEPRESS
**DOI：** 10.5220/0006112800500060
**原文：** `Sabri2017_SemanticPattern_FloorPlanRetrieval_ICPRAM.pdf`（11 页）
**状态：** ✅ 已下载原文（开放获取）
**技术路线：** CBR + 精确/非精确图匹配 | gnn 的符号化对照

> 出处说明：本文是 Napps 2021 的参考文献 **[14]**（Althoff 一线：CBR + 精确/非精确图匹配）。
> 延伸版本：Sabri et al. (2018) *Semi-automated Testing of an Architectural Floor Plan Retrieval Framework...*, LNCS 10857:169–189（DOI 10.1007/978-3-319-93647-5_10，Springer 付费）。

## 中文摘要

在建筑平面图的概念设计中，建筑师至今仍普遍采用传统的纸笔方法绘制概念平面图，并在纸质参考集中检索相似平面图。本文提出一个**端到端的完整系统**，帮助建筑师在早期设计阶段检索相似的平面平面图。工作有三方面贡献：

1. **适配并对比三种检索相似平面图的方法**：**案例推理（CBR）**、**精确图匹配（exact graph matching）** 与**非精确图匹配（inexact graph matching）**；
2. 通过实验**探测各检索方法的计算极限（computational limits）**；
3. 在考虑计算极限的前提下，用建筑师构造的更真实测试案例做**定性分析**。

为可视化结果，作者集成了此前实现的 Web 用户界面的增强版。**定性分析表明：精确图匹配（基于 VF2 算法）在大多数测试案例上总体优于另外两种方法。** 该工作的新颖之处在于：**在建筑平面图检索这一领域中，把 CBR、精确图匹配与非精确图匹配统一到同一个系统里**。

**关键词：** 图匹配；子图匹配；图同构；建筑平面图；案例推理；模式识别

## 原文摘要（English）

> ... In this paper we present a complete end-to-end system that helps architects to retrieve similar floor plans in early design phases. This work makes a three-fold contribution. Firstly, we have adapted three state of the art techniques to retrieve the similar floor plans: case-based reasoning (CBR), exact graph matching, and inexact graph matching. ... The qualitative analysis showed that the exact graph matching gives in general better results for a majority of test cases, as compared to other two methods. The novelty of our approach is that it combines CBR, exact, and inexact graph matching in one system in the domain of retrieval of architectural floor plans.

## 与本项目的关系

**最直接的方法论对应**：该文在同一系统内对比 **CBR / 精确图匹配 / 非精确图匹配** 三种检索路线，与本项目阶段 2 对比 **statistics / gnn / hybrid** 三条路线高度类似；其"在真实检索场景下比较不同相似性机制并做定量+定性评价"的做法，可为本项目的基准设计与结论表述提供参照。此外，它把"符号化图匹配"与"案例特征检索"结合，也与本项目"结构体系先验 + 类内特征检索"两阶段思路相通。
