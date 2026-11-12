# -*- coding: utf-8 -*-
"""Fix 3.2 lead-in sentence after moving the metric explanation to 3.1."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261112_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_32lead.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
OLD = "其数学定义如式(4)所示（d为特征向量的维度）："
NEW = ("在此基础上，按曼哈顿距离、欧氏距离与余弦相似度三种度量计算案例相似程度，"
       "其定义如式(4)所示（d为特征向量的维度）：")


def qn(tag):
    return "{%s}%s" % (W, tag)


def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("t")))


def rpr_from(p):
    for run in p.findall(qn("r")):
        if run.find(qn("rPr")) is not None:
            return run.find(qn("rPr"))
    return None


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])
    hits = [p for p in root.iter(qn("p")) if para_text(p).strip() == OLD]
    if len(hits) != 1:
        raise RuntimeError("lead matched %d" % len(hits))
    p = hits[0]
    rpr = rpr_from(p)
    for child in list(p):
        if child.tag != qn("pPr"):
            p.remove(child)
    r = etree.SubElement(p, qn("r"))
    if rpr is not None:
        r.append(etree.fromstring(etree.tostring(rpr)))
    t = etree.SubElement(r, qn("t"))
    t.set("{%s}space" % XML, "preserve")
    t.text = NEW
    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("3.2 lead fixed")


if __name__ == "__main__":
    main()
