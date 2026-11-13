# -*- coding: utf-8 -*-
"""Table 7 v3: per-cell tcMar=40, widen method col, 0.7458 -> 0.75."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261112_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_table7_layout.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"

GRID = [1560, 500, 680, 500, 680, 559]  # total 4479
MARGIN = "40"


def qn(tag):
    return "{%s}%s" % (W, tag)


def cell_text(tc):
    return "".join(t.text or "" for t in tc.iter(qn("t")))


def main():
    shutil.copyfile(BAK, DOCX)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])

    target = None
    for tbl in root.iter(qn("tbl")):
        tr = tbl.find(qn("tr"))
        if tr is not None and "NDCG@3" in "".join(cell_text(tc) for tc in tr.findall(qn("tc"))):
            target = tbl
            break
    if target is None:
        raise RuntimeError("table 7 not found")

    tblPr = target.find(qn("tblPr"))
    layout = tblPr.find(qn("tblLayout"))
    if layout is None:
        layout = etree.SubElement(tblPr, qn("tblLayout"))
    layout.set(qn("type"), "fixed")

    grid = target.find(qn("tblGrid"))
    for c, w in zip(grid.findall(qn("gridCol")), GRID):
        c.set(qn("w"), str(w))

    for tr in target.findall(qn("tr")):
        for idx, tc in enumerate(tr.findall(qn("tc"))):
            if idx >= len(GRID):
                continue
            tcPr = tc.find(qn("tcPr"))
            if tcPr is None:
                tcPr = etree.Element(qn("tcPr"))
                tc.insert(0, tcPr)
            tcW = tcPr.find(qn("tcW"))
            if tcW is None:
                tcW = etree.Element(qn("tcW"))
                tcPr.insert(0, tcW)
            tcW.set(qn("w"), str(GRID[idx]))
            tcW.set(qn("type"), "dxa")

            tcMar = tcPr.find(qn("tcMar"))
            if tcMar is None:
                tcMar = etree.Element(qn("tcMar"))
                anchor = None
                for tag in ("textDirection", "tcFitText", "vAlign"):
                    anchor = tcPr.find(qn(tag))
                    if anchor is not None:
                        break
                if anchor is not None:
                    anchor.addprevious(tcMar)
                else:
                    tcPr.append(tcMar)
            for side in ("top", "left", "bottom", "right"):
                e = tcMar.find(qn(side))
                if e is None:
                    e = etree.SubElement(tcMar, qn(side))
                e.set(qn("w"), "0" if side in ("top", "bottom") else MARGIN)
                e.set(qn("type"), "dxa")

            if cell_text(tc).strip() == "0.7458":
                for t in tc.iter(qn("t")):
                    if (t.text or "").strip() == "0.7458":
                        t.text = "0.75"
                        t.set("{%s}space" % XML, "preserve")

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("table7 v3 applied")


if __name__ == "__main__":
    main()
