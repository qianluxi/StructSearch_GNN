# -*- coding: utf-8 -*-
"""Revise section 2.4 (Z-score) and append Larose & Larose (2014) as [23]."""
import copy
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_zscore.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"

INTRO = ("上述11维特征包含构件比例、面积规模、统计分布及空间距离等不同类型的数值变量，其量纲和取值范围存在差异。"
         "在基于距离的相似性计算中，特征的数值尺度会影响其对距离的贡献，若直接使用原始特征值进行比较，"
         "数值尺度较大的维度可能主导距离计算[22-23]。为降低不同特征尺度对检索结果的影响，本文采用Z-score标准化"
         "（标准分数标准化）对各维特征分别进行中心化与尺度缩放，使标准化后的各维特征均值为0、标准差为1[23]。"
         "其计算公式为：")
EXPLAIN = ("其中，xᵢ为原始第i维特征值，μᵢ和σᵢ分别为案例库参考集合中该维特征的均值和标准差；当标准差为0或接近0时，"
           "取1以避免除零。标准化仅利用案例库的特征数值计算统计量，不使用专家评分或其他相关性标注信息。需要指出，"
           "Z-score标准化用于消除不同特征的量纲和数值尺度差异，并不意味着各维特征具有相同的工程重要性。"
           "第5.2节通过对比实验分析标准化对结构相似性检索性能的影响。")
LAROSE = ("[23] LAROSE D T, LAROSE C D. Discovering Knowledge in Data: An Introduction to Data Mining[M]. "
          "2nd ed. Hoboken, NJ: John Wiley & Sons, 2014.")


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
    paras = list(root.iter(qn("p")))

    # 1) replace 2.4 intro and explanation
    for anchor, text in [("上述11维特征量纲不同", INTRO),
                         ("其中μᵢ、σᵢ分别为案例库中第i维特征", EXPLAIN)]:
        hits = [p for p in paras if para_text(p).startswith(anchor)]
        if len(hits) != 1:
            raise RuntimeError("anchor %r matched %d" % (anchor, len(hits)))
        set_para_text(hits[0], text)

    # 2) append Larose as [23] after [22] LIAO
    ref22 = None
    for p in paras:
        if para_text(p).startswith("[22] LIAO"):
            ref22 = p
            break
    if ref22 is None:
        raise RuntimeError("[22] LIAO not found")
    newp = copy.deepcopy(ref22)
    set_para_text(newp, LAROSE)
    ref22.addnext(newp)

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("2.4 revised; [23] Larose added")


if __name__ == "__main__":
    main()
