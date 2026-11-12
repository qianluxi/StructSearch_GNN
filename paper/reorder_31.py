# -*- coding: utf-8 -*-
"""Reorder 3.1: metrics-concept paragraph first, then Stage-1 algorithm; keep inline OMML."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261112_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_reorder31.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML = "http://www.w3.org/XML/1998/namespace"

PARA1 = ("案例检索通常将案例表示为特征向量，并通过向量之间的距离或相似度衡量案例的接近程度。相似性度量是影响"
         "候选案例排序的重要设计选择，既有案例推理（CBR）研究对相关度量方法及混合清晰/模糊特征的比较进行了综述[22]。"
         "常见的度量方式包括曼哈顿距离、欧氏距离与余弦相似度。其中，曼哈顿距离通过累加各维特征差值的绝对值衡量"
         "两个案例的差异；欧氏距离通过各维特征差值的平方和的平方根衡量两个案例在多维空间中的直线距离。二者均属于"
         "距离度量，数值越小表示案例越相近。余弦相似度则通过两个特征向量夹角的余弦值衡量其方向一致性，主要反映"
         "不同维度特征标准化偏离模式的相似程度，数值越大表示案例越相似。与前两种距离度量不同，余弦相似度关注"
         "向量方向，而不直接衡量向量之间的绝对距离。由于本文采用Z-score方法对11维结构描述子进行标准化，各维分量"
         "可能为负，因此余弦相似度的理论取值范围为[−1,1]，而非仅为[0,1]。此时，余弦相似度反映的是各维特征相对于"
         "样本均值的标准化偏离方向及其组合模式，并不完全等同于原始结构特征比例的相似程度。本文在后续案例检索实验中"
         "比较上述三种度量方式，其数学定义见第3.2节式(4)。")

P2_BEFORE = ("在第一阶段的结构体系识别中，本文采用“以检索代分类”的策略，不依赖任何训练模型：以库内其余图纸的"
             "标准化描述子为参照样本，计算查询描述子与全部参照样本的余弦相似度，取相似度最高的3个近邻，并以其中"
             "余弦相似度最高的最近邻的结构体系类型作为预测类型[5]。需要说明三点：其一，本文报告的识别准确率采用"
             "库内留一检索式口径，即每张图纸在查询时以其余105张图纸为参照，不涉及独立测试集；其二，为反映3个近邻"
             "对预测类别的相对支持程度，本文对3个近邻的余弦相似度sⱼ（j=1,2,3）进行Softmax归一化，得到3个非负且"
             "总和为1的权重：")
P2_AFTER = ("，j=1,2,3；将属于预测类别的近邻权重相加，作为该类别的“近邻相对支持度”；其三，该支持度仅反映当前"
            "3个近邻之间的相对相似度分布，并未经过独立验证集的概率校准，不代表预测类别实际正确的概率，仅用于结果"
            "排序与人工复核参考。由于预测类型由最近邻直接确定，不涉及多数投票，因此不存在1:1:1平局问题。在106张"
            "图纸上，该方法的库内留一检索式结构体系识别Top-1准确率为87.7%，表明统计特征向量对结构体系已具备较强"
            "的判别能力。")


def qn(tag):
    return "{%s}%s" % (W, tag)


def qnm(tag):
    return "{%s}%s" % (M, tag)


def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("t")))


def rpr_from(p):
    for run in p.findall(qn("r")):
        if run.find(qn("rPr")) is not None:
            return run.find(qn("rPr"))
    return None


def clear_text(p):
    for child in list(p):
        if child.tag != qn("pPr"):
            p.remove(child)


def text_run(p, txt, rpr):
    r = etree.SubElement(p, qn("r"))
    if rpr is not None:
        r.append(etree.fromstring(etree.tostring(rpr)))
    t = etree.SubElement(r, qn("t"))
    t.set("{%s}space" % XML, "preserve")
    t.text = txt


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])
    paras = list(root.iter(qn("p")))

    p_cls = [p for p in paras if para_text(p).startswith("结构体系识别采用")]
    p_met = [p for p in paras if "三种度量方式进行比较" in para_text(p)]
    if len(p_cls) != 1 or len(p_met) != 1:
        raise RuntimeError("paragraphs: cls=%d met=%d" % (len(p_cls), len(p_met)))
    p_cls, p_met = p_cls[0], p_met[0]

    # extract existing Softmax OMML from classifier paragraph
    omath = p_cls.find(qnm("oMath"))
    if omath is None:
        raise RuntimeError("Softmax oMath not found in 3.1 classifier paragraph")
    omath = etree.fromstring(etree.tostring(omath))
    rpr_cls = rpr_from(p_cls)
    rpr_met = rpr_from(p_met)

    # rebuild classifier paragraph: text + OMML + text
    clear_text(p_cls)
    text_run(p_cls, P2_BEFORE, rpr_cls)
    p_cls.append(omath)
    text_run(p_cls, P2_AFTER, rpr_cls)

    # rebuild metrics paragraph
    clear_text(p_met)
    text_run(p_met, PARA1, rpr_met)

    # reorder: metrics paragraph before classifier paragraph
    p_cls.addprevious(p_met)

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("3.1 reordered: metrics first, algorithm second")


if __name__ == "__main__":
    main()
