# Thinking with Containers: A Multi-Agent Retrieval Approach for the Case-Based Semantic Search of Architectural Designs

**作者：** Viktor Ayzenshtadt, Christoph Langenhan, Saqib Bukhari, **Klaus-Dieter Althoff**, Frank Petzold, Andreas Dengel
**出处：** ICAART 2016 (8th Int. Conf. on Agents and Artificial Intelligence), pp. 149–156, SCITEPRESS
**DOI：** 10.5220/0005683801490156
**原文：** `Ayzenshtadt2016_ThinkingWithContainers_ICAART.pdf`（8 页）
**状态：** ✅ 已下载原文（开放获取）
**技术路线：** 多智能体 CBR 编排 | hybrid

> 出处说明：Napps 2021 参考文献 [14] 所依赖的 **MetisCBR** 系统来源论文（Althoff / Ayzenshtadt 一线）。

## 中文摘要

为了在设计概念化过程中提供"有用信息"的检索，并借助**分布式人工智能**的优势，可以采用一种把**检索相关任务与知识维护任务**分配给**自主工作且具备案例自学习能力的智能体（agent）及其智能体组**的方法。

本文提出面向建筑设计领域的**分布式检索系统 MetisCBR**：智能体以**组（containers，容器）** 为单位协同工作，处理以**语义描述模型 Semantic Fingerprint（语义指纹）** 构造的用户查询。该方法的核心目标是：为建筑师构建一个相当实用的检索工具——通过把**案例推理（CBR）** 与**多智能体（multi-agent）** 方法相结合，在大型建筑设计集合中获得有价值、有帮助的检索结果。

**关键词：** 案例推理；多智能体系统；语义指纹；建筑设计检索；容器

## 原文摘要（English）

> To provide the retrieval of information, that can be considered useful during the design conceptualization process, with advantages of distributed artificial knowledge, an approach, that distributes retrieval-related and knowledge maintaining tasks among autonomously working and case-based self-learning agents and agent groups, can be used. In this work we present the distributed retrieval system MetisCBR for the architectural design domain, where agents work in groups (containers) on resolving of user queries built with a semantic description model Semantic Fingerprint. The main aim of our approach is to carry out a basis for a considerable retrieval tool for architects, where the combination of case-based reasoning and multi-agent methods helps to achieve valuable and helpful search results in a comprehensive building design collection.

## 与本项目的关系

是 Sabri 等（2017，本目录 [14]）中 **MetisCBR** 检索路线的来源，用"多智能体 + 语义指纹 + CBR"组织建筑设计的案例检索。其"把检索拆成可组合的知识源/智能体"的架构，可为本项目未来把"类型识别 / 统计检索 / 图检索 / 加权融合"拆成独立可编排模块提供借鉴；也说明在建筑检索领域，**分布式编排 + 案例检索**是成熟方向。
