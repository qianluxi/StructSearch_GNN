# High-Level Feature Aggregation for Fine-Grained Architectural Floor Plan Retrieval

**作者：** Divya Sharma, Chiranjoy Chattopadhyay
**出处：** IET Computer Vision, 2018
**DOI：** 10.1049/iet-cvi.2017.0581
**原文：** `SharmaThesis_ContentBasedFloorPlanRetrieval.pdf`（作者博士论文，151 页，含并通过该论文内容）
**状态：** 📄 以博士论文替代（原期刊论文未获取）
**技术路线：** 高层语义特征加权检索 | statistics

## 中文摘要

随着房地产行业的高速发展，用于找房/房源展示的在线平台大量增加；相比描述性文字，**按例查询（query-by-example）** 始终是更受青睐的检索方式。平面图是反映某一层建筑结构的基本 2D 表示。作者提出一个在按例查询范式下检索**相似建筑平面图**的框架：首先提出一种从建筑平面图中提取**高层语义特征**的新算法；进而提出基于**特征加权和**的**细粒度检索**，允许在检索时对某些特征赋予更高权重。实验在含 510 张平面图的公开数据集（ROBIN）上进行，并与现有最新方法比较，本方法在**定性和定量上均优于**它们。

**关键词：** 平面图检索；按例查询；高层语义特征；特征融合；细粒度检索

## 原文摘要（English）

> Due to the massive growth of the real-estate industry, the number of online platforms designed for finding homes/furnished properties has increased. Instead of descriptive words, query-by-example is always a preferred retrieval method. ... The authors propose a framework for retrieving similar architectural floor plans under the query-by-example paradigm. They propose a novel algorithm to extract high-level semantic features from an architectural floor plan. Fine-grained retrieval using a weighted sum of the features is proposed ... Experiments were performed on a publicly available dataset containing 510 floor plans ... The proposed method outperforms others both qualitatively and quantitatively.

## 与本项目的关系

**细粒度（fine-grained）检索 + 特征加权融合**与本项目"类内相似排序 + 特征组消融（类型/面积/空间）"直接对应。其"按需给某些特征更高权重"的思路，可为本项目后续对 11 维描述子分组加权提供参考。
