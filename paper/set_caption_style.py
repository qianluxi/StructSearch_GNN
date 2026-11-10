# -*- coding: utf-8 -*-
"""Set the '图名' (caption) style to 9pt SimHei (east asia) + Times New Roman."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_caption_style.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def qn(tag):
    return "{%s}%s" % (W, tag)


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    styles = etree.fromstring(parts["word/styles.xml"])

    target = None
    for st in styles.iter(qn("style")):
        name = st.find(qn("name"))
        if name is not None and name.get(qn("val")) == "图名":
            target = st
            break
    if target is None:
        raise RuntimeError("图名 style not found")

    rPr = target.find(qn("rPr"))
    if rPr is None:
        rPr = etree.SubElement(target, qn("rPr"))
    # remove existing rFonts/sz
    for tag in ("rFonts", "sz", "szCs"):
        old = rPr.find(qn(tag))
        if old is not None:
            rPr.remove(old)
    fonts = etree.SubElement(rPr, qn("rFonts"))
    fonts.set(qn("ascii"), "Times New Roman")
    fonts.set(qn("hAnsi"), "Times New Roman")
    fonts.set(qn("eastAsia"), "SimHei")
    sz = etree.SubElement(rPr, qn("sz"))
    sz.set(qn("val"), "18")  # 9 pt (小五)
    szcs = etree.SubElement(rPr, qn("szCs"))
    szcs.set(qn("val"), "18")

    parts["word/styles.xml"] = etree.tostring(styles, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("caption style set: 9pt SimHei + Times New Roman")


if __name__ == "__main__":
    main()
