# -*- coding: utf-8 -*-
"""Rewrite 3.2: explain three metrics, show all three formulas (OMML), ranking + cosine range."""
import copy
import os
import shutil
import zipfile

import latex2mathml.converter as l2m
import mathml2omml
from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_metrics3.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML = "http://www.w3.org/XML/1998/namespace"
Wn, Mn = "{%s}" % W, "{%s}" % M

INTRO = ("案例检索中的相似性度量是影响候选排序的重要设计选择，既有CBR研究对相关度量方法及混合清晰/模糊特征的"
         "比较进行了综述[22]。本文对曼哈顿距离、欧氏距离与余弦相似度三种度量方式进行比较。其中，曼哈顿距离通过"
         "累加各维特征差值的绝对值衡量两个案例的差异；欧氏距离通过各维特征差值的平方和的平方根衡量两个案例在"
         "多维空间中的直线距离；二者均属于距离度量，数值越小表示案例越相近。余弦相似度则通过两个特征向量夹角的"
         "余弦值衡量其方向一致性，主要反映不同维度特征变化模式的相似程度，数值越大表示案例越相似；与前两种距离"
         "度量不同，余弦相似度关注向量方向，而不直接衡量向量之间的绝对距离。设查询案例和候选案例经标准化后的"
         "特征向量分别为q与x（维度均为d），三种度量的定义如式(4)所示：")

NOTE = ("其中，d为特征向量的维度；曼哈顿距离与欧氏距离均按数值由小到大的顺序判断案例相似程度（数值越小越相似），"
        "余弦相似度则按数值由大到小的顺序判断（数值越大越相似）。需要说明，由于本文11维结构描述子经过Z-score标准化，"
        "各维分量可能为负，余弦相似度的理论取值范围为[−1,1]而非仅[0,1]；此时余弦相似度比较的是各维特征相对于样本"
        "均值的标准化偏离方向及其组合模式，并不完全等同于比较原始结构特征的比例关系。")

FORMULA = (r"\begin{split} "
           r"\text{曼哈顿距离：}\ D_1(q,x) &= \sum_{i=1}^{d}|q_i-x_i| \\ "
           r"\text{欧氏距离：}\ D_2(q,x) &= \sqrt{\sum_{i=1}^{d}(q_i-x_i)^2} \\ "
           r"\text{余弦相似度：}\ S_{\cos}(q,x) &= "
           r"\frac{\sum_{i=1}^{d}q_i x_i}{\sqrt{\sum_{i=1}^{d}q_i^2}\sqrt{\sum_{i=1}^{d}x_i^2}} "
           r"\end{split}")

OLD_LEAD = "最终检索函数定义为候选集内按距离升序取前K个："
NEW_LEAD = ("最终检索函数定义为候选集内按相似程度排序取前K个（曼哈顿距离与欧氏距离按升序，余弦相似度按降序），"
            "以距离度量为例：")


def qn(tag):
    return "{%s}%s" % (W, tag)


def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("t")))


def omml(latex):
    ml = l2m.convert(latex)
    frag = mathml2omml.convert(ml)
    r = etree.fromstring(('<root xmlns:m="%s">%s</root>' % (M, frag)).encode("utf-8"))
    return r[0]


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

    # 1) intro paragraph (contains 三种度量方式进行比较)
    intro = [p for p in paras if "三种度量方式进行比较" in para_text(p)]
    if len(intro) != 1:
        raise RuntimeError("intro matched %d" % len(intro))
    intro_p = intro[0]
    rpr_intro = rpr_from(intro_p)
    clear_text(intro_p)
    text_run(intro_p, INTRO, rpr_intro)

    # 2) replace equation (4)
    eq4 = [p for p in paras if para_text(p).strip() == "(4)"]
    if len(eq4) != 1:
        raise RuntimeError("eq (4) matched %d" % len(eq4))
    eq4_p = eq4[0]
    rpr_eq = rpr_from(eq4_p)
    old_num_run = eq4_p.findall(qn("r"))
    clear_text(eq4_p)
    eq4_p.append(omml(FORMULA))
    text_run(eq4_p, "　　(4)", rpr_eq)

    # 3) insert NOTE paragraph after eq(4)
    note_p = copy.deepcopy(intro_p)  # same body style
    for child in list(note_p):
        if child.tag != qn("pPr"):
            note_p.remove(child)
    text_run(note_p, NOTE, rpr_intro)
    eq4_p.addnext(note_p)

    # 4) update lead-in to TopK
    lead = [p for p in paras if OLD_LEAD in para_text(p)]
    if len(lead) != 1:
        raise RuntimeError("lead matched %d" % len(lead))
    lp = lead[0]
    rpr_lead = rpr_from(lp)
    clear_text(lp)
    text_run(lp, NEW_LEAD, rpr_lead)

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("3.2 metrics rewritten; note inserted")


if __name__ == "__main__":
    main()
