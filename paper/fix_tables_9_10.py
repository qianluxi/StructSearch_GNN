# -*- coding: utf-8 -*-
"""Tidy headers/layout of Table 9 (by type) and Table 10 (same/cross project)."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261112_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_tables9_10.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"

MARGIN = "40"


def qn(tag):
    return "{%s}%s" % (W, tag)


def cell_text(tc):
    return "".join(t.text or "" for t in tc.iter(qn("t"))).strip()


def ensure(parent, tag, before=None):
    el = parent.find(qn(tag))
    if el is None:
        el = etree.Element(qn(tag))
        if before is not None:
            before.addprevious(el)
        else:
            parent.append(el)
    return el


def set_cell(tc, width, margin=MARGIN):
    tcPr = tc.find(qn("tcPr"))
    if tcPr is None:
        tcPr = etree.Element(qn("tcPr"))
        tc.insert(0, tcPr)
    tcW = tcPr.find(qn("tcW"))
    if tcW is None:
        tcW = etree.Element(qn("tcW"))
        tcPr.insert(0, tcW)
    tcW.set(qn("w"), str(width))
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
        e.set(qn("w"), "0" if side in ("top", "bottom") else margin)
        e.set(qn("type"), "dxa")


def fix_table(tbl, grid, header_sz=None, split=None):
    tblPr = tbl.find(qn("tblPr"))
    layout = ensure(tblPr, "tblLayout")
    layout.set(qn("type"), "fixed")
    for c, w in zip(tbl.find(qn("tblGrid")).findall(qn("gridCol")), grid):
        c.set(qn("w"), str(w))
    for tr in tbl.findall(qn("tr")):
        for idx, tc in enumerate(tr.findall(qn("tc"))):
            if idx < len(grid):
                set_cell(tc, grid[idx])
    if header_sz or split:
        header = tbl.find(qn("tr"))
        for tc in header.findall(qn("tc")):
            key = cell_text(tc)
            parts = (split or {}).get(key, (key,))
            p = tc.find(qn("p"))
            runs = p.findall(qn("r"))
            if not runs:
                continue
            run = runs[0]
            rpr = run.find(qn("rPr"))
            for child in list(run):
                if child.tag != qn("rPr"):
                    run.remove(child)
            if header_sz and rpr is not None:
                sz = rpr.find(qn("sz"))
                if sz is None:
                    sz = etree.SubElement(rpr, qn("sz"))
                sz.set(qn("val"), header_sz)
            for i, part in enumerate(parts):
                t = etree.SubElement(run, qn("t"))
                t.set("{%s}space" % XML, "preserve")
                t.text = part
                if i < len(parts) - 1:
                    etree.SubElement(run, qn("br"))
            for extra in runs[1:]:
                p.remove(extra)


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])

    n = 0
    for tbl in root.iter(qn("tbl")):
        tr = tbl.find(qn("tr"))
        if tr is None:
            continue
        hdr = "".join(cell_text(tc) for tc in tr.findall(qn("tc")))
        if "统计特征向量" in hdr and "类型" in hdr:
            fix_table(tbl, [1000, 1900, 1579])
            n += 1
        elif "子集" in hdr and "NDCG@3" in hdr:
            fix_table(
                tbl,
                [1120, 1240, 700, 700, 719],
                header_sz="14",
                split={
                    "NDCG@3": ("NDCG", "@3"),
                    "NDCG@5": ("NDCG", "@5"),
                    "Recall@5": ("Recall", "@5"),
                },
            )
            n += 1
    if n != 2:
        raise RuntimeError("expected 2 tables, fixed %d" % n)

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("tables 9/10 fixed")


if __name__ == "__main__":
    main()
