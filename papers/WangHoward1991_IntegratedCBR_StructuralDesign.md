# Integrated Case-Based Reasoning for Structural Design

**作者：** Jenmu Wang（王人牧）, H. Craig Howard
**出处：** CIFE Technical Report No. 58, Center for Integrated Facility Engineering, Stanford University, 1991
**URL：** https://purl.stanford.edu/gd764wd4870
**原文：** `WangHoward1991_IntegratedCBR_StructuralDesign.pdf`（扫描版，130 页，无文字层）
**状态：** ✅ 已下载原文
**技术路线：** 结构设计 CBR 框架 | statistics / hybrid

## 中文摘要

近期结构工程设计的知识系统多聚焦于**与案例无关的知识（case-independent knowledge）**，即设计的一般推理规则——这方面虽取得长足进展，但仍亟需开发能利用每一个实质性结构设计中所蕴含的丰富知识的系统；另一方面，以往面向数据库的设计工作主要建立在"**知识贫乏**"的解数据库之上，只是把传统工程手册的解换成了数字数据。挑战在于：如何捕获并应用结构工程师惯用的、**与案例相关的知识（case-dependent knowledge）**。

本文研究**同时使用**"与案例相关"和"与案例无关"知识来实现知识化结构设计系统的途径。所实现的系统 **DDIS** 在**黑板（blackboard）框架**下把案例推理与案例无关推理结合起来：与案例无关的模块表示问题域与问题求解策略的抽象知识；与案例相关的模块用 CBR 技术把以往设计的知识迁移到当前设计任务。文中指出，以往设计解可直接用于非常相似的新设计，而以往设计**计划（plans）** 则可用于引导"表面相似度较低"的设计。DDIS 在钢梁-柱设计与输电杆塔锚固底板设计两个应用上得到验证。

**关键词：** 案例推理；结构设计；黑板模型；DDIS；案例相关/无关知识

## 原文摘要（English）

> Recent knowledge-based expert systems for structural engineering design have focused on case-independent knowledge ... there is still a significant need to develop systems to take advantage of the wealth of knowledge contained in every substantial structural design. ... This thesis/technical report examines the new approach of implementing knowledge-based structural design systems using both case-dependent and case-independent knowledge. The resulting system, **DDIS**, combines case-based reasoning with case-independent reasoning in a blackboard framework. ... Two structural design applications have been built using DDIS: structural steel beam-columns and anchor base plates for electrical transmission poles.

## 与本项目的关系

结构工程中"案例推理（CBR）"的**奠基性工作之一**，提出了"用历史设计案例辅助新设计"的框架——正是本项目"历史图纸相似案例检索"的工程思想源头。其"案例表征（representation）→ 检索 → 适配"三段式，与本项目"特征构建 → 阶段 1 判型 → 阶段 2 类内检索"结构同构；文中"以往计划可引导表面相似度较低的设计"的观点，也呼应本项目"类型先验 + 类内细粒度检索"的设计。
