# -*- coding: utf-8 -*-
"""Rewrite 3.1: explain Softmax (inline formula), rename to 近邻相对支持度, match code."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_softmax.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"

NEW = ("结构体系识别采用“以检索代分类”的策略，不依赖任何训练模型：以库内其余图纸的标准化描述子为参照样本，"
       "计算查询描述子与全部参照样本的余弦相似度，取相似度最高的3个近邻，并以其中余弦相似度最高的最近邻的"
       "结构体系类型作为预测类型[5]。需要说明三点：其一，本文报告的识别准确率采用库内留一检索式口径，"
       "即每张图纸在查询时以其余105张图纸为参照，不涉及独立测试集；其二，为反映3个近邻对预测类别的相对"
       "支持程度，本文对3个近邻的余弦相似度sⱼ（j=1,2,3）作Softmax归一化，wⱼ = exp(sⱼ) / Σₖ₌₁³ exp(sₖ)，"
       "得到3个非负且总和为1的权重；将属于预测类别的近邻权重相加，作为该类别的“近邻相对支持度”；其三，"
       "该支持度仅反映当前3个近邻之间的相对相似度分布，并未经过独立验证集的概率校准，不代表预测类别实际"
       "正确的概率，仅用于结果排序与人工复核参考。由于预测类型由最近邻直接确定，不涉及多数投票，因此不存在"
       "1:1:1平局问题。在106张图纸上，该方法的库内留一检索式结构体系识别Top-1准确率为87.7%，表明统计特征"
       "向量对结构体系已具备较强的判别能力。")


def qn(tag):
    return "{%s}%s" % (W, tag)


def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("t")))


def set_para_text(p, text):
    rpr = None
    for run in p.findall(qn("r")):
        if run.find(qn("rPr")) is not None:
            rpr = run.find(qn("rPr"))
            break
    for run in list(p.findall(qn("r"))):
        p.remove(run)
    run = etree.SubElement(p, qn("r"))
    if rpr is not None:
        run.append(etree.fromstring(etree.tostring(rpr)))
    t = etree.SubElement(run, qn("t"))
    t.set("{%s}space" % XML, "preserve")
    t.text = text


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])
    hits = [p for p in root.iter(qn("p")) if para_text(p).startswith("结构体系识别采用")]
    if len(hits) != 1:
        raise RuntimeError("3.1 paragraph matched %d" % len(hits))
    if any(el.tag == qn("commentReference") for el in hits[0].iter()):
        raise RuntimeError("target paragraph has a comment")
    set_para_text(hits[0], NEW)
    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("3.1 rewritten")


if __name__ == "__main__":
    main()
