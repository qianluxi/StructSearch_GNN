# Supporting the Building Design Process with Graph-based Methods Using Centrally Coordinated Federated Databases

**作者：** Johannes Roith, Christoph Langenhan, Frank Petzold（TU Munich）
**出处：** Visualization in Engineering, 2017, 5: 20（SpringerOpen，开放获取）
**DOI：** 10.1186/s40327-017-0057-y
**原文：** `Roith2017_GraphBasedDesign_FederatedDB_VisEng.pdf`（17 页）
**状态：** ✅ 已下载原文（开放获取）
**技术路线：** 图指纹 / 图匹配 | gnn 的符号化对照

> 出处说明：本文是 Napps 2021 的参考文献 **[13]**。

## 中文摘要

**背景：** 技术发展与全球化工作流程改变了建造过程，但现有的数字化语义建筑模型已无法充分表达现代建筑项目日益增长的复杂性；"基于智能体（agent）+ 基于图（graph）"方法的潜力（如能耗计算、空间检索策略）尚未被充分利用。

**方法：** 系统中，用户在多点触控设备上**手绘建筑局部的概念草图**，系统分析该草图并据此查询一个**联邦数据库系统**——由建筑信息模型服务器、图数据库 **neo4j** 与内容管理系统 **mediaTUM** 组成。客户端通过与**中央协调器（Coordinator）** 服务通信，持续接收并更新平面图列表。

**结果/结论：** 系统用**语义化的建筑平面图"指纹"（floorplan fingerprints）** 进行查询——指纹被形式化为**图**并编码为统一 schema（**AGraphML**），以表示空间构型并执行**图匹配**。由于图匹配计算代价高，协调器需分析查询、拆分不同指纹与元数据，并将其分派给专门的**微服务（agent）** 处理，最后合并并可视化结果。采用多智能体的好处在于：可重组来自异构数据源的数据，并支持依据查询属性和预定义规则选择不同执行策略。为应对复杂动态查询和动态结果，作者还设计了**缓存框架**，在判断查询等价性时考虑图表示相似度。

**关键词：** 检索策略；建筑信息模型（BIM）；联邦数据库；图匹配；智能体

## 原文摘要（English）

> **Background:** Technological developments and globalized working processes have transformed the building process. ... The potential of combining agent-based and graph-based methods ... is not fully exploited. **Methods:** ... users search for building floorplans ... by creating conceptual hand-drawn sketches ... and used to query a federated database system comprising a building information model server, the graph database neo4j and the content management system mediaTUM. ... **Results:** Our federated database system is queried using semantic building floorplan fingerprints, which are formalized as graphs and encoded in a common schema, like our AGraphML, to represent spatial configurations and perform graph matching. ... **Conclusions:** ... a caching framework was developed which takes the similarity measures of the graph-based building representation into account when determining query equality.

## 与本项目的关系

展示了"**以图表示建筑平面构型 + 图匹配检索 + 多服务/规则调度**"的完整工程实现（Metis 项目），与本项目"用图/特征表示结构并按相似度检索"同属"结构案例检索"家族。其"依查询复杂度动态选择检索策略"的设计，可作为本项目 future work 中"按类型/场景自适应切换 statistics/gnn/hybrid 路线"的工程参考；其"缓存 + 相似度"思路也提示本项目可缓存跨查询的索引相似度。
