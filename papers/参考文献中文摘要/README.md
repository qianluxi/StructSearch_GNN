# papers/ — 文献资料索引（按技术路线分类）

本目录收集与 "结构统计特征 + 两阶段相似检索" 相关的研究，**每篇均有原文 PDF 与中文摘要**。

- 共 **20 篇**：原文 PDF **20 篇**，中文摘要 **20 篇**。
- 每个摘要文件（`*.md`）含：题录、**技术路线/对应路线**、中文摘要、原文摘要（英文，若有）、以及"与本项目关系"。
- 分类依据为**检索技术路线**，并标注其与本项目 **statistics / gnn / hybrid** 三条路线的关系。
- 更新日期：2026-09-16

---

## 路线总览：与本项目三条路线的关系

| 本项目路线 | 含义 | 本目录中与之对应的文献路线 |
|---|---|---|
| **statistics** | 11 维结构统计描述子的无学习检索 | 路线一（统计/几何特征描述子）为主；路线三（CBR 相似度度量）为其理论骨架 |
| **gnn** | 图神经网络嵌入检索 | 路线二（图结构/图匹配）是其**符号化对照**（同一"图"数据，用图匹配而非学习嵌入） |
| **hybrid** | GNN 嵌入 + 统计描述子拼接 / 多方法编排 | 路线三（CBR 多方法编排）、路线四（学习式与多方法基线） |

---

## 路线一、统计 / 几何特征描述子检索 → **statistics**

直接同源于本项目"把结构/图纸压缩为全局统计或几何描述子再检索"的做法。

| # | 文献 | 原文 PDF | 中文摘要 | 对应路线 |
|---|---|---|---|---|
| 1 | **de las Heras et al. (2014)** *Runlength Histogram Image Signature for Perceptual Retrieval of Architectural Floor Plans*, LNCS | [PDF](delasHeras2014_RunlengthHistogram_FloorPlans.pdf) | [摘要](delasHeras2014_RunlengthHistogram_FloorPlans.md) | statistics |
| 2 | **Rodrigues et al. (2017)** *Clustering of Architectural Floor Plans: A Comparison of Shape Representations*, Automation in Construction 80:48–65 | [PDF](Rodrigues2017_Clustering_FloorPlans.pdf) | [摘要](Rodrigues2017_Clustering_FloorPlans.md) | statistics |
| 3 | **Ahmed et al. (2013)** *Automatic Analysis and Sketch-Based Retrieval of Architectural Floor Plans*, Pattern Recognition Letters 35:91–100 | [PDF](Ahmed2013_Analysis_SketchRetrieval_FloorPlans.pdf) | [摘要](Ahmed2013_Analysis_SketchRetrieval_FloorPlans.md) | statistics |
| 4 | **Sharma & Chattopadhyay (2018)** *High-Level Feature Aggregation for Fine-Grained Architectural Floor Plan Retrieval*, IET Computer Vision | [PDF(博士论文)](SharmaThesis_ContentBasedFloorPlanRetrieval.pdf) | [摘要](SharmaChattopadhyay2018_FineGrained_FloorPlanRetrieval.md) | statistics |
| 5 | **Ling, Luo, Zhou & Jiang (2025)** *Survey of Architectural Floor Plan Retrieval Technology Based on 3ST Features*, AI (MDPI) 6(4):67 | [PDF](Survey2025_FloorPlanRetrieval_MDPI.pdf) | [摘要](Survey2025_FloorPlanRetrieval_MDPI.md) | statistics |

## 路线二、图结构 / 图匹配检索 → **gnn 的符号化对照**

与本项目 **gnn** 路线使用"同一份图数据"，但检索手段是**符号化子图匹配**而非学习式嵌入——是本项目结论（GNN 未超越统计特征）的重要对照面。

| # | 文献 | 原文 PDF | 中文摘要 | 对应路线 |
|---|---|---|---|---|
| 6 | **Sabri, Bayer, Ayzenshtadt, Bukhari, Althoff & Dengel (2017)** *Semantic Pattern-based Retrieval of Architectural Floor Plans with Case-based and Graph-based Searching Techniques...*, ICPRAM:50–60 | [PDF](Sabri2017_SemanticPattern_FloorPlanRetrieval_ICPRAM.pdf) | [摘要](Sabri2017_SemanticPattern_FloorPlanRetrieval_ICPRAM.md) | gnn（对照） |
| 7 | **Roith, Langenhan & Petzold (2017)** *Supporting the Building Design Process with Graph-based Methods Using Centrally Coordinated Federated Databases*, Visualization in Engineering 5:20 | [PDF](Roith2017_GraphBasedDesign_FederatedDB_VisEng.pdf) | [摘要](Roith2017_GraphBasedDesign_FederatedDB_VisEng.md) | gnn（对照） |

## 路线三、案例推理（CBR）与相似度度量 → **statistics / hybrid 的理论骨架**

提供"检索 = 相似度度量 + 案例表示"的方法学框架，是 statistics 与 hybrid 两条路线共同的理论基础。

| # | 文献 | 原文 PDF | 中文摘要 | 对应路线 |
|---|---|---|---|---|
| 8 | **Aamodt & Plaza (1994)** *Case-Based Reasoning: Foundational Issues...*, AI Communications 7(1):39–59 | [PDF](AamodtPlaza1994_CBR_Foundations.pdf) | [摘要](AamodtPlaza1994_CBR_Foundations.md) | statistics / hybrid |
| 9 | **Börner (1993)** *Structural Similarity as Guidance in Case-Based Design* | [PDF](Borner1993_StructuralSimilarity_CBR.pdf) | [摘要](Borner1993_StructuralSimilarity_CBR.md) | hybrid |
| 10 | **Liao, Zhang & Mount (1998)** *Similarity Measures for Retrieval in Case-Based Reasoning Systems*, Applied AI 12(4):267–288 | [PDF](Liao1998_SimilarityMeasures_CBR.pdf) | [摘要](Liao1998_SimilarityMeasures_CBR.md) | statistics / hybrid |
| 11 | **Wang & Howard (1991)** *Integrated Case-Based Reasoning for Structural Design*, Stanford CIFE TR058 | [PDF](WangHoward1991_IntegratedCBR_StructuralDesign.pdf) | [摘要](WangHoward1991_IntegratedCBR_StructuralDesign.md) | statistics / hybrid |
| 12 | **Sriram, Maher & Fenves (1985)** *Knowledge-Based Expert Systems in Structural Design*, Computers & Structures 20(1–3):1–9 | [PDF](SriramMaherFenves1985_KBES_StructuralDesign.pdf) | [摘要](SriramMaherFenves1985_KBES_StructuralDesign.md) | statistics（背景） |
| 13 | **Maher (1987)** *Expert Systems for Structural Design*, J. Computing in Civil Engineering 1(4):270–278 | [PDF](Maher1987_ExpertSystems_StructuralDesign.pdf) | [摘要](Maher1987_ExpertSystems_StructuralDesign.md) | statistics（前身） |
| 14 | **Maher & Gómez de Silva Garza (1996)** *Developing Case-Based Reasoning for Structural Design*, IEEE Expert 11(3):42–52 | [PDF](MaherGomez1996_CBR_StructuralDesign.pdf) | [摘要](MaherGomez1996_CBR_StructuralDesign.md) | statistics / hybrid |
| 15 | **Maher & Gómez de Silva Garza (1997)** *Case-based reasoning in design*, IEEE Expert 12(2):34–41 | [PDF](MaherGomez1997_CBR_in_Design.pdf) | [摘要](MaherGomez1997_CBR_in_Design.md) | statistics / hybrid |
| 16 | **Soibelman & Peña-Mora (2000)** *Distributed Multi-Reasoning Mechanism to Support Conceptual Structural Design*, JSE 126(6):733–742 | [PDF](SoibelmanPenaMora2000_MRAM_ConceptualStructuralDesign.pdf) | [摘要](SoibelmanPenaMora2000_MRAM_ConceptualStructuralDesign.md) | hybrid |
| 17 | **Napps, Pawlowski & König (2021)** *BIM-based Variant Retrieval...CBR & Pattern Matching*, ISARC 38:435–442 | [PDF](Napps2021_BIM_VariantRetrieval_CBR.pdf) | [摘要](Napps2021_BIM_VariantRetrieval_CBR.md) | hybrid |

## 路线四、学习式 / 多方法编排与基线对照 → **gnn / hybrid**

涉及学习式检索、无学习 vs 深度对照、多智能体编排，为本项目 gnn 与 hybrid 提供基线和方法论参照。

| # | 文献 | 原文 PDF | 中文摘要 | 对应路线 |
|---|---|---|---|---|
| 18 | **Zhang et al. (2006)** *BP-based Intelligent Case Retrieval Method of High-rise Structure Form Selection*, IEEE | [PDF](Zhang2006_BP_CaseRetrieval_HighriseStructureForm.pdf) | [摘要](Zhang2006_BP_CaseRetrieval_HighriseStructureForm.md) | gnn / hybrid 对照 |
| 19 | **van Engelenburg et al. (2023)** *SSIG: A Visually-Guided Graph Edit Distance for Floor Plan Similarity*, ICCV-W | [PDF](vanEngelenburg2023_SSIG_FloorPlan.pdf) | [摘要](vanEngelenburg2023_SSIG_FloorPlan.md) | statistics（支持无学习） |
| 20 | **Ayzenshtadt, Langenhan, Bukhari, Althoff, Petzold & Dengel (2016)** *Thinking with Containers: A Multi-Agent Retrieval Approach...*, ICAART:149–156 | [PDF](Ayzenshtadt2016_ThinkingWithContainers_ICAART.pdf) | [摘要](Ayzenshtadt2016_ThinkingWithContainers_ICAART.md) | hybrid |

---

## 附：与本项目结论的呼应

- **支持 statistics 路线为主**：van Engelenburg (2023) 论证"无学习框架即可媲美深度方法"；Liao (1998) 指出"相似度度量的选择决定检索效果"；Börner (1993) 提出"粗筛 + 精匹配"两级结构——共同支撑本项目"11D 统计描述子优于 GNN"的结论。
- **呼应 gnn 路线**：Sabri (2017) 中精确图匹配(VF2)优于非精确图匹配与 CBR；Roith (2017) 显示图匹配计算代价高、需多服务编排——说明"图结构"路线在检索中普遍面临"结构感知 vs 计算/泛化"的权衡，与本项目 GNN 跨项目泛化差的发现一致。
- **呼应 hybrid 路线**：Soibelman (2000) 与 Ayzenshtadt (2016) 都用"多方法/多智能体编排"整合异构检索；Napps (2021) 用 CBR + 模式匹配组合——为本项目 hybrid（GNN 嵌入 + 统计描述子拼接）及 future work 的"路线自适应调度"提供参照。

> 说明：本目录仅收录已获取原文的文献；曾引用但无法获取原文的条目（Multimedia CBR 1994、Rafiq 1989、Ayzenshtadt ICCBR 2016）已移除。未使用盗版镜像，也未绕过付费墙。
