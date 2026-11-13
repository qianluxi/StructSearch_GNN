# -*- coding: utf-8 -*-
"""Table 7: header cells -> 8pt with clean line breaks."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261112_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_table7_headers.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"

SPLIT = {
    "NDCG@3": ("NDCG", "@3"),
    "Recall@3": ("Recall", "@3"),
    "NDCG@5": ("NDCG", "@5"),
    "Recall@5": ("Recall", "@5"),
    "MeanScore@3": ("Mean", "Score", "@3"),
}
HEADER_SZ = "14"  # 7 pt


def qn(tag):
    return "{%s}%s" % (W, tag)


def cell_text(tc):
    return "".join(t.text or "" for t in tc.iter(qn("t"))).strip()


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])

    target = None
    for tbl in root.iter(qn("tbl")):
        tr = tbl.find(qn("tr"))
        if tr is not None and "NDCG" in "".join(cell_text(tc) for tc in tr.findall(qn("tc"))):
            target = tbl
            break
    if target is None:
        raise RuntimeError("table 7 not found")

    header = target.find(qn("tr"))
    for tc in header.findall(qn("tc")):
        key = cell_text(tc)
        parts_ = SPLIT.get(key, (key,))
        p = tc.find(qn("p"))
        runs = p.findall(qn("r"))
        if not runs:
            continue
        run = runs[0]
        rpr = run.find(qn("rPr"))
        for child in list(run):
            if child.tag != qn("rPr"):
                run.remove(child)
        if rpr is not None:
            sz = rpr.find(qn("sz"))
            if sz is None:
                sz = etree.SubElement(rpr, qn("sz"))
            sz.set(qn("val"), HEADER_SZ)
        for i, part in enumerate(parts_):
            t = etree.SubElement(run, qn("t"))
            t.set("{%s}space" % XML, "preserve")
            t.text = part
            if i < len(parts_) - 1:
                etree.SubElement(run, qn("br"))
        for extra in runs[1:]:
            p.remove(extra)

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("table7 headers 8pt")


if __name__ == "__main__":
    main()
