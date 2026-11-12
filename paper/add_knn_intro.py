# -*- coding: utf-8 -*-
"""Introduce KNN: add abbreviation at first mention (3.3) and a brief explanation in 5.1."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261112_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_knn.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

EDITS = [
    ("以质心K近邻（k=6）确定节点连接",
     "以质心K近邻（K-nearest neighbors，KNN，k=6）确定节点连接"),
    ("构图采用KNN，k=6",
     "构图采用KNN（每个构件节点与其平面距离最近的6个节点相连），k=6"),
]


def qn(tag):
    return "{%s}%s" % (W, tag)


def replace_in_para(p, old, new):
    ts = list(p.iter(qn("t")))
    full = "".join(t.text or "" for t in ts)
    if old not in full:
        return False
    for t in ts:
        if old in (t.text or ""):
            t.text = t.text.replace(old, new)
            return True
    ts[0].text = full.replace(old, new)
    for t in ts[1:]:
        t.text = ""
    return True


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])
    done = 0
    for old, new in EDITS:
        for p in root.iter(qn("p")):
            if replace_in_para(p, old, new):
                done += 1
                break
    if done != len(EDITS):
        raise RuntimeError("applied %d/%d edits" % (done, len(EDITS)))
    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("KNN introduced: %d edits" % done)


if __name__ == "__main__":
    main()
