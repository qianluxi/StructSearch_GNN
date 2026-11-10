# -*- coding: utf-8 -*-
"""Re-add the missing comment id 7 (section 4.3 bibliography note)."""
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_readd_c7.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"

ANCHOR = "本文采用信息检索评价指标进行检索排序性能分析"
TEXT = ("【书目校勘·待校核】4.3 已把 [10] 限定为 NDCG 的方法来源（Järvelin & Kekäläinen），"
        "MeanScore@K 与 Recall@K 为本文自定义指标。请核对。")


def qn(tag):
    return "{%s}%s" % (W, tag)


def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("t")))


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}

    # append definition if missing
    comments = etree.fromstring(parts["word/comments.xml"])
    ids = [c.get(qn("id")) for c in comments.findall(qn("comment"))]
    if "7" not in ids:
        c = etree.SubElement(comments, qn("comment"))
        c.set(qn("id"), "7")
        c.set(qn("author"), "AI 审稿建议")
        c.set(qn("initials"), "AI")
        c.set(qn("date"), "2026-11-09T00:00:00Z")
        p = etree.SubElement(c, qn("p"))
        r = etree.SubElement(p, qn("r"))
        t = etree.SubElement(r, qn("t"))
        t.set("{%s}space" % XML, "preserve")
        t.text = TEXT
        parts["word/comments.xml"] = etree.tostring(comments, xml_declaration=True, encoding="UTF-8", standalone=True)

    root = etree.fromstring(parts["word/document.xml"])
    hits = [p for p in root.iter(qn("p")) if ANCHOR in para_text(p)]
    if len(hits) != 1:
        raise RuntimeError("anchor matched %d" % len(hits))
    target = hits[0]
    # avoid duplicate
    if any(el.get(qn("id")) == "7" for el in target.iter(qn("commentRangeStart"))):
        print("already present")
    else:
        start = etree.Element(qn("commentRangeStart"))
        start.set(qn("id"), "7")
        pPr = target.find(qn("pPr"))
        if pPr is not None:
            pPr.addnext(start)
        else:
            target.insert(0, start)
        end = etree.Element(qn("commentRangeEnd"))
        end.set(qn("id"), "7")
        target.append(end)
        run = etree.SubElement(target, qn("r"))
        ref = etree.SubElement(run, qn("commentReference"))
        ref.set(qn("id"), "7")

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("comment 7 re-added")


if __name__ == "__main__":
    main()
