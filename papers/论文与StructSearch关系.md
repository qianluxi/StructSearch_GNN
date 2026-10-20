# 论文与 StructSearch 关系梳理

> 本目录收录 19 篇与"建筑/结构图纸检索"相关的论文（mono 版）。整体可归为三条脉络：
> 1. **CBR 与相似性度量方法论** —— 为 StructSearch 提供理论依据；
> 2. **结构设计中的知识/案例复用** —— StructSearch 的领域来源；
> 3. **建筑平面图/结构图纸的图表示与检索** —— StructSearch 的方法谱系。
>
> StructSearch 位于三者交叉处：**以工程语义特征，对结构图纸做相似案例检索**。

---

## 一、CBR 与相似性度量（方法论）

- **AamodtPlaza1994_CBR_Foundations-mono**：CBR 奠基论文（检索–复用–修正–保留循环）。→ 说明"用历史案例支持新问题"这一范式，正是 StructSearch 存在的意义；StructSearch 实现其中"检索(Retrieve)"环节。

- **Liao1998_SimilarityMeasures_CBR-mono**：CBR 检索相似性度量综述 + 混合（精确/模糊）度量。→ 为 StructSearch 的距离度量选择（欧氏/曼哈顿/余弦、加权）提供方法论依据。

- **Borner1993_StructuralSimilarity_CBR-mono**：以"结构相似性"（最具体的公共结构 + 变换规则）指导 CBR 的设计适应。→ 与 StructSearch 同名的"结构相似性"但含义不同：其侧重图形公共子结构；可对照 StructSearch 的统计描述子路线。

- **Ayzenshtadt2016_ThinkingWithContainers_ICAART-mono**：MetisCBR——多智能体、分布式 CBR 检索建筑方案（基于语义指纹）。→ 同谱系的"分布式检索架构"，是 Sabri/Sharma 前身之一。

---

## 二、结构设计中的知识复用（领域）

- **Maher1987_ExpertSystems_StructuralDesign-mono**：结构设计专家系统（HI-RISE/FLODER/LOCATOR/STRUPLE）。→ 结构设计早期决策计算机支持的源头，说明该领域"复用设计经验"的价值。

- **SriramMaherFenves1985_KBES_StructuralDesign-mono**：基于知识的结构设计专家系统（KBES）概念框架。→ "知识驱动结构设计"的早期代表，是 StructSearch 动机的祖先。

- **WangHoward1991_IntegratedCBR_StructuralDesign-mono**：集成的基于案例的结构设计（扫描件，无可提取文本）。→ 结构设计中集成 CBR 的早期工作，领域先例。

- **MaherGomez1996_CBR_StructuralDesign-mono**：CBR 用于结构设计（CaseCAD/CADsyn；FBS 表示；从真实工程公司建库时遇到的困难）。→ 最直接的结构设计 CBR 先例，支持"结构案例检索"，也印证其数据获取之难。

- **MaherGomez1997_CBR_in_Design-mono**：设计中 CBR 综述（设计辅助 vs 自动化；复杂案例表示；检索/适应）。→ 设计 CBR 的理论框架，支撑 StructSearch 的设计科学定位。

- **SoibelmanPenaMora2000_MRAM_ConceptualStructuralDesign-mono**：M-RAM 分布式多推理机制支持概念结构设计（生成适配的既有方案）。→ 概念结构设计中"复用/改编既有方案"的早期多智能体实现，是 StructSearch 领域的直接前导。

---

## 三、平面图/结构图纸的图表示与检索（方法谱系）

- **Ahmed2013_Analysis_SketchRetrieval_FloorPlans-mono**：a.SCatch——平面图分析（墙/符号/房间检测）+ 草图检索 + 图匹配。→ 同谱系：图纸→图→匹配；StructSearch 以 DXF 解析替代其图像分析。

- **Roith2017_GraphBasedDesign_FederatedDB_VisEng-mono**：Coordinator 中间件 + 联邦数据库（BIM+Neo4j+mediaTUM）+ 微服务 + 加权排序 + 查询缓存 + 交互草图。→ 同谱系的"系统/架构"侧；其**加权排序**与**查询缓存**思路可借鉴，但非可比算法。

- **Sabri2017_SemanticPattern_FloorPlanRetrieval_ICPRAM-mono**：Archistant——语义指纹（7 种）+ 三种检索（MetisCBR / VF2 精确图匹配 / 索引非精确）。→ 直接相关工作：图/指纹检索；证明显式图匹配（VF2）可最优；提示 StructSearch 补拓扑特征。

- **delasHeras2014_RunlengthHistogram_FloorPlans-mono**：游程长度直方图图像签名做**外观/风格**检索。→ 低层外观方法，与 StructSearch 的工程语义特征形成对照（"外观相似 ≠ 结构相似"）。

- **Rodrigues2017_Clustering_FloorPlans-mono**：平面图形状表示方法比较（点距离/转向函数/网格/切线距离）+ 层次聚类。→ 描述子表示与比较的方法论参考，可对照 StructSearch 的特征设计。

- **Napps2021_BIM_VariantRetrieval_CBR-mono**：基于 BIM/IFC 的方案变体检索（CBR + 模式匹配，图 + 几何）。→ 现代 BIM 版"方案检索"；StructSearch 是其在结构图纸上的对应。

- **Survey2025_FloorPlanRetrieval_MDPI-mono**：3ST（语义/空间/形状/纹理）特征检索综述，含小波/傅里叶描述子。→ 最新综述，用于给 StructSearch 的特征体系定位、补充相关工作。

- **vanEngelenburg2023_SSIG_FloorPlan-mono**：SSIG（IoU + 图编辑距离 GED）度量平面图结构相似性，无学习、可大规模排序（RPLAN）。→ 最相关的现代方法之一：图像 + 图结合度量结构相似性；"简单结构度量可比深度方法"与 StructSearch 结论呼应。

- **SharmaThesis_ContentBasedFloorPlanRetrieval-mono**：博士论文——拓扑谱嵌入 + RLM/RDM、CNN、语义特征 RAS/CAR/FCR + 加权融合、草图检索（Cyclic GAN）。→ 最详尽的直接相关工作；"手工语义特征优于深度特征""先粗后细两阶段"与 StructSearch 高度一致；其 **RAS 邻接特征最强**提示 StructSearch 补拓扑。

---

## 小结

| 脉络 | 代表文献 | 对 StructSearch 的意义 |
| --- | --- | --- |
| CBR / 相似性度量 | AamodtPlaza1994、Liao1998、Borner1993、Ayzenshtadt2016 | 理论依据：为何"检索相似案例"有意义；相似性度量的选择方法 |
| 结构设计知识复用 | Maher1987、SriramMaherFenves1985、WangHoward1991、MaherGomez1996/1997、SoibelmanPenaMora2000 | 领域来源：结构设计中复用/改编既有方案 |
| 图纸图表示与检索 | Ahmed2013、Roith2017、Sabri2017、delasHeras2014、Rodrigues2017、Napps2021、Survey2025、vanEngelenburg2023、SharmaThesis | 方法谱系：图纸→图/特征→相似检索；提供对比、评价与改进方向 |

**核心结论**：StructSearch_v2 的定位是"**以工程语义统计描述子 + 两阶段（判型→类内检索）对结构图纸做相似案例检索**"。上表三条脉络分别回答了它的**理论合理性（CBR）**、**领域价值（结构设计复用）**与**方法位置（图纸图检索）**。
