# -*- coding: utf-8 -*-
"""Replace 3.3 with expanded term explanations; add retrieval-scope note in 5.1."""
import copy
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261112_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_revise_33.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"

SEC33 = ("为考察深度表征学习方法在该任务上的表现，本文以图神经网络（Graph Neural Network，下文简称GNN）作为"
         "对照方法[6-7]：将结构平面构件集合建模为无向图，构件为节点，以质心K近邻（K-nearest neighbors，KNN，"
         "k=6）确定节点连接，节点间欧氏距离作为边属性；节点特征为构件类型独热编码、面积、跨度、壁厚与长细比等"
         "9维。图编码器采用3层GraphSAGE卷积[7]，通过对节点邻域进行采样与聚合更新节点特征表示。随后采用全局均值"
         "池化，将图中所有节点的特征表示按维度取平均，汇总为代表整张结构平面的图级表示；再通过线性映射生成128维"
         "图嵌入，即用于表征整张结构平面的128维数值向量。在此基础上，分类输出层产生3个结构体系类别的logits，即"
         "Softmax归一化前的未归一化分类得分。模型以结构类型分类为训练目标，采用交叉熵损失函数衡量分类输出与真实"
         "类别标签之间的差异，并通过反向传播更新网络参数。图纸样本按80%/20%划分训练集与验证集，其中训练集用于"
         "模型参数学习，验证集用于监测模型表现并选择验证准确率最高的检查点；随机种子设为42，以控制数据划分和训练"
         "过程中的主要随机因素，提高实验可复现性。GNN检索以128维图嵌入的余弦相似度排序；Hybrid方法将128维GNN嵌入"
         "与标准化11维统计特征向量拼接为139维向量后作余弦检索。需要强调，GNN在本文中仅作为深度对照方法，其训练"
         "目标为结构类型分类而非案例相似度排序，两阶段检索框架本身不依赖任何训练模型。")

NOTE = ("需要说明，GNN的80%/20%训练/验证划分仅用于结构类型分类的训练与检查点选择，检索评价未按该划分裁剪："
        "索引覆盖案例库全部106张图纸，15个查询及其候选池均取自该库，且未剔除训练阶段已见过的图纸。因此，"
        "检索基准属于库内检索口径，其结果用于比较各方法在库内检索中的相对表现，而非估计对未见图纸的泛化能力。")


def qn(tag):
    return "{%s}%s" % (W, tag)


def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("t")))


def rpr_from(p):
    for run in p.findall(qn("r")):
        if run.find(qn("rPr")) is not None:
            return run.find(qn("rPr"))
    return None


def set_paragraph_text(p, text, rpr):
    for child in list(p):
        if child.tag != qn("pPr"):
            p.remove(child)
    r = etree.SubElement(p, qn("r"))
    if rpr is not None:
        r.append(etree.fromstring(etree.tostring(rpr)))
    t = etree.SubElement(r, qn("t"))
    t.set("{%s}space" % XML, "preserve")
    t.text = text


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])

    p33 = [p for p in root.iter(qn("p")) if para_text(p).startswith("为考察深度表征学习方法在该任务上的表现")]
    if len(p33) != 1:
        raise RuntimeError("3.3 paragraph matched %d" % len(p33))
    p33 = p33[0]
    set_paragraph_text(p33, SEC33, rpr_from(p33))

    p51 = [p for p in root.iter(qn("p")) if para_text(p).startswith("全部实验在CPU上完成")]
    if len(p51) != 1:
        raise RuntimeError("5.1 paragraph matched %d" % len(p51))
    p51 = p51[0]
    note = copy.deepcopy(p51)
    for child in list(note):
        if child.tag != qn("pPr"):
            note.remove(child)
    r = etree.SubElement(note, qn("r"))
    rpr = rpr_from(p51)
    if rpr is not None:
        r.append(etree.fromstring(etree.tostring(rpr)))
    t = etree.SubElement(r, qn("t"))
    t.set("{%s}space" % XML, "preserve")
    t.text = NOTE
    p51.addnext(note)

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("3.3 replaced; 5.1 note added")


if __name__ == "__main__":
    main()
