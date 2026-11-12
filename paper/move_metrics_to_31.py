# -*- coding: utf-8 -*-
"""Move the three-metric explanation from 3.2 to 3.1 (first appearance of cosine similarity)."""
import copy
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261112_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_move_metrics.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"

EXPLAIN = ("案例检索中的相似性度量是影响候选排序的重要设计选择，既有CBR研究对相关度量方法及混合清晰/模糊特征的"
           "比较进行了综述[22]。本文对曼哈顿距离、欧氏距离与余弦相似度三种度量方式进行比较。其中，曼哈顿距离通过"
           "累加各维特征差值的绝对值衡量两个案例的差异；欧氏距离通过各维特征差值的平方和的平方根衡量两个案例在"
           "多维空间中的直线距离；二者均属于距离度量，数值越小表示案例越相近。余弦相似度则通过两个特征向量夹角的"
           "余弦值衡量其方向一致性，主要反映不同维度特征变化模式的相似程度，数值越大表示案例越相似；与前两种距离"
           "度量不同，余弦相似度关注向量方向，而不直接衡量向量之间的绝对距离。需要说明，由于本文11维结构描述子"
           "经过Z-score标准化，各维分量可能为负，余弦相似度的理论取值范围为[−1,1]而非仅[0,1]；此时余弦相似度"
           "比较的是各维特征相对于样本均值的标准化偏离方向及其组合模式，并不完全等同于比较原始结构特征的比例关系。"
           "三种度量的数学定义见第3.2节式(4)。")

LEAD = "其数学定义如式(4)所示（d为特征向量的维度）："


def qn(tag):
    return "{%s}%s" % (W, tag)


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

    # 1) insert explanation after 3.1 classifier paragraph
    p31 = [p for p in paras if para_text(p).startswith("结构体系识别采用")]
    if len(p31) != 1:
        raise RuntimeError("3.1 paragraph matched %d" % len(p31))
    p31 = p31[0]
    rpr = rpr_from(p31)
    newp = copy.deepcopy(p31)
    clear_text(newp)
    text_run(newp, EXPLAIN, rpr)
    p31.addnext(newp)

    # 2) replace 3.2 explanation with short lead-in
    p32 = [p for p in paras if "三种度量方式进行比较" in para_text(p)]
    if len(p32) != 1:
        raise RuntimeError("3.2 explanation matched %d" % len(p32))
    p32 = p32[0]
    rpr32 = rpr_from(p32)
    clear_text(p32)
    text_run(p32, LEAD, rpr32)

    # 3) remove the NOTE paragraph ("其中，d为特征向量的维度...")
    notes = [p for p in paras if para_text(p).startswith("其中，d为特征向量的维度")]
    for n in notes:
        n.getparent().remove(n)

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("moved explanation to 3.1; 3.2 lead-in set; note removed:", len(notes))


if __name__ == "__main__":
    main()
