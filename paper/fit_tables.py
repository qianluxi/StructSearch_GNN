# -*- coding: utf-8 -*-
"""Scale all table widths to fit the two-column text width (7.9 cm)."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_table_fit.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
TARGET_TWIPS = int(7.9 * 567)  # 7.9 cm


def qn(tag):
    return "{%s}%s" % (W, tag)


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])

    for tbl in root.iter(qn("tbl")):
        grid = tbl.find(qn("tblGrid"))
        cols = grid.findall(qn("gridCol")) if grid is not None else []
        widths = [int(c.get(qn("w")) or 0) for c in cols]
        total = sum(widths)
        if total <= 0:
            # fall back to first-row tcW
            tr = tbl.find(qn("tr"))
            widths = []
            if tr is not None:
                for tc in tr.findall(qn("tc")):
                    tcW = tc.find(qn("tcPr") + "/" + qn("tcW"))
                    widths.append(int(tcW.get(qn("w")) or 0) if tcW is not None else 0)
            total = sum(widths)
            if total <= 0:
                continue
            if grid is None:
                grid = etree.Element(qn("tblGrid"))
                tbl.insert(1, grid)
            for w in widths:
                gc = etree.SubElement(grid, qn("gridCol"))
                gc.set(qn("w"), str(w))
            cols = grid.findall(qn("gridCol"))

        new = [max(1, round(w * TARGET_TWIPS / total)) for w in widths]
        # fix rounding so sum == target
        diff = TARGET_TWIPS - sum(new)
        if new:
            new[-1] += diff
        for gc, w in zip(cols, new):
            gc.set(qn("w"), str(w))

        # update cell widths (dxa) proportionally
        for tr in tbl.findall(qn("tr")):
            for tc, w in zip(tr.findall(qn("tc")), new):
                tcPr = tc.find(qn("tcPr"))
                if tcPr is None:
                    continue
                tcW = tcPr.find(qn("tcW"))
                if tcW is not None and tcW.get(qn("type"), "dxa") == "dxa":
                    tcW.set(qn("w"), str(w))

        tblPr = tbl.find(qn("tblPr"))
        if tblPr is not None:
            tblW = tblPr.find(qn("tblW"))
            if tblW is None:
                tblW = etree.SubElement(tblPr, qn("tblW"))
            tblW.set(qn("type"), "dxa")
            tblW.set(qn("w"), str(TARGET_TWIPS))

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("tables scaled to", TARGET_TWIPS, "twips (7.9cm)")


if __name__ == "__main__":
    main()
