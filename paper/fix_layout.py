# -*- coding: utf-8 -*-
"""Normalize figure/table layout in the paper DOCX."""
import os
import re
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_layout_fix.docx"
FIG2 = r"F:\project\PyProject\GNN\StructSearch_v2\paper\figures\fig2_methods_portrait.png"
FIG3 = r"F:\project\PyProject\GNN\StructSearch_v2\paper\figures\fig3_ablation_portrait.png"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML = "http://www.w3.org/XML/1998/namespace"

# target display width (cm) per media filename
WIDTH_CM = {
    "image1.png": 8.5,
    "image2.png": 10.0,
    "image3.png": 10.0,
    "image4.png": 11.0,
    "image5.png": 12.0,
    "image6.png": 16.0,
}
EMU_PER_CM = 360000


def qn(tag):
    return "{%s}%s" % (W, tag)


def get_or_add(parent, tag):
    el = parent.find(tag)
    if el is None:
        el = etree.SubElement(parent, tag)
    return el


def set_para_format(p, center=True, before=None, after=None):
    pPr = p.find(qn("pPr"))
    if pPr is None:
        pPr = etree.Element(qn("pPr"))
        p.insert(0, pPr)
    # remove existing jc/spacing then re-add
    for tag in ("jc", "spacing"):
        old = pPr.find(qn(tag))
        if old is not None:
            pPr.remove(old)
    if center:
        jc = etree.SubElement(pPr, qn("jc"))
        jc.set(qn("val"), "center")
    if before is not None or after is not None:
        sp = etree.SubElement(pPr, qn("spacing"))
        if before is not None:
            sp.set(qn("before"), str(before))
        if after is not None:
            sp.set(qn("after"), str(after))
        sp.set(qn("line"), "240")
        sp.set(qn("lineRule"), "auto")


def set_cell_margins(tbl, top=0, bottom=0, left=57, right=57):
    tblPr = tbl.find(qn("tblPr"))
    if tblPr is None:
        tblPr = etree.Element(qn("tblPr"))
        tbl.insert(0, tblPr)
    old = tblPr.find(qn("tblCellMar"))
    if old is not None:
        tblPr.remove(old)
    mar = etree.SubElement(tblPr, qn("tblCellMar"))
    for edge, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = etree.SubElement(mar, qn(edge))
        e.set(qn("w"), str(val))
        e.set(qn("type"), "dxa")


def set_fixed_layout(tbl):
    tblPr = tbl.find(qn("tblPr"))
    if tblPr is None:
        tblPr = etree.Element(qn("tblPr"))
        tbl.insert(0, tblPr)
    old = tblPr.find(qn("tblLayout"))
    if old is not None:
        tblPr.remove(old)
    lay = etree.SubElement(tblPr, qn("tblLayout"))
    lay.set(qn("type"), "fixed")


def set_row_cant_split(tbl):
    for tr in tbl.findall(qn("tr")):
        trPr = tr.find(qn("trPr"))
        if trPr is None:
            trPr = etree.Element(qn("trPr"))
            tr.insert(0, trPr)
        if trPr.find(qn("cantSplit")) is None:
            etree.SubElement(trPr, qn("cantSplit"))


def set_cell_valign_center(tbl):
    for tc in tbl.iter(qn("tc")):
        tcPr = tc.find(qn("tcPr"))
        if tcPr is None:
            tcPr = etree.Element(qn("tcPr"))
            tc.insert(0, tcPr)
        old = tcPr.find(qn("vAlign"))
        if old is not None:
            tcPr.remove(old)
        va = etree.SubElement(tcPr, qn("vAlign"))
        va.set(qn("val"), "center")


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}

    # replace wide figures with portrait versions
    with open(FIG2, "rb") as f:
        parts["word/media/image2.png"] = f.read()
    with open(FIG3, "rb") as f:
        parts["word/media/image3.png"] = f.read()

    from PIL import Image
    import io
    ratios = {}
    for name in WIDTH_CM:
        im = Image.open(io.BytesIO(parts["word/media/" + name]))
        ratios[name] = im.width / im.height

    root = etree.fromstring(parts["word/document.xml"])
    rels = etree.fromstring(parts["word/_rels/document.xml.rels"])
    relmap = {r.get("Id"): os.path.basename(r.get("Target")) for r in rels}

    WPn = "{%s}" % WP
    An = "{%s}" % A
    Rn = "{%s}" % R

    drawings = [el for el in root.iter() if el.tag in (WPn + "inline", WPn + "anchor")]
    print("drawings:", len(drawings))
    for i, d in enumerate(drawings):
        blip = d.find(".//" + An + "blip")
        rid = blip.get(Rn + "embed") if blip is not None else None
        media = relmap.get(rid)
        if media not in WIDTH_CM:
            raise RuntimeError("unexpected media %r" % media)
        cx = int(WIDTH_CM[media] * EMU_PER_CM)
        cy = int(cx / ratios[media])
        ext = d.find(WPn + "extent")
        ext.set("cx", str(cx))
        ext.set("cy", str(cy))
        for aext in d.iter(An + "ext"):
            aext.set("cx", str(cx))
            aext.set("cy", str(cy))

        if d.tag == WPn + "anchor":
            d.tag = WPn + "inline"
            for child in list(d):
                t = child.tag
                if t in (WPn + "simplePos", WPn + "positionH", WPn + "positionV") or t.startswith(WPn + "wrap"):
                    d.remove(child)
            for k in list(d.attrib):
                del d.attrib[k]
            for a in ("distT", "distB", "distL", "distR"):
                d.set(a, "0")

        # center + spacing on the containing paragraph
        p = d.getparent()
        while p is not None and p.tag != qn("p"):
            p = p.getparent()
        if p is not None:
            set_para_format(p, center=True, before=160, after=40)
        print("  %d %s cx=%.2fcm cy=%.2fcm" % (i, media, cx / EMU_PER_CM, cy / EMU_PER_CM))

    # captions: center + consistent spacing
    for p in root.iter(qn("p")):
        text = "".join(t.text or "" for t in p.iter(qn("t"))).strip()
        if re.match(r"^(图|表)\s*\d+", text):
            set_para_format(p, center=True, before=40, after=140)
        elif text.startswith("注："):
            set_para_format(p, center=True, before=0, after=120)

    # normalize tables
    tables = list(root.iter(qn("tbl")))
    for tbl in tables:
        set_cell_margins(tbl)
        set_fixed_layout(tbl)
        set_row_cant_split(tbl)
        set_cell_valign_center(tbl)

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("saved:", DOCX)


if __name__ == "__main__":
    main()
