# -*- coding: utf-8 -*-
"""Add a brief explanation of t-SNE in the discussion paragraph."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261112_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_tsne.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"

OLD = "图4的t-SNE可视化[11]直观显示了三类结构在GNN嵌入空间中的分离情况，仅作为辅助观察，不作为评价指标。"
NEW = ("图4的t-SNE可视化[11]直观显示了三类结构在GNN嵌入空间中的分离情况。其中，t-SNE（t-分布随机邻域嵌入，"
       "t-distributed stochastic neighbor embedding）是一种非线性降维方法，通过在高维与低维空间中保持样本邻域"
       "的概率分布，将高维嵌入映射到二维平面，以便观察样本的聚类与分离；该图仅作为辅助观察，不作为评价指标。")


def qn(tag):
    return "{%s}%s" % (W, tag)


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])

    done = False
    for p in root.iter(qn("p")):
        ts = list(p.iter(qn("t")))
        full = "".join(t.text or "" for t in ts)
        if OLD not in full:
            continue
        for t in ts:
            if OLD in (t.text or ""):
                t.text = (t.text or "").replace(OLD, NEW)
                done = True
                break
        if not done:
            ts[0].text = full.replace(OLD, NEW)
            ts[0].set("{%s}space" % XML, "preserve")
            for t in ts[1:]:
                t.text = ""
            done = True
        break
    if not done:
        raise RuntimeError("target sentence not found")

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("t-SNE explanation added")


if __name__ == "__main__":
    main()
