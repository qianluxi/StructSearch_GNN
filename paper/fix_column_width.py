# -*- coding: utf-8 -*-
"""Fit figures to the two-column layout; make the large 图6 span both columns."""
import copy
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_column_fix.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
EMU = 360000

WPn = "{%s}" % WP
An = "{%s}" % A
Rn = "{%s}" % R

# image6 spans both columns; others single column
WIDTH_CM = {
    "image1.png": 7.8,
    "image2.png": 7.8,
    "image3.png": 7.8,
    "image4.png": 7.8,
    "image5.png": 7.8,
    "image6.png": 16.5,
}


def qn(tag):
    return "{%s}%s" % (W, tag)


def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("t")))


def ancestor_para(el):
    p = el.getparent()
    while p is not None and p.tag != qn("p"):
        p = p.getparent()
    return p


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}

    from PIL import Image
    import io
    ratios = {}
    for name in WIDTH_CM:
        im = Image.open(io.BytesIO(parts["word/media/" + name]))
        ratios[name] = im.width / im.height

    root = etree.fromstring(parts["word/document.xml"])
    rels = etree.fromstring(parts["word/_rels/document.xml.rels"])
    relmap = {r.get("Id"): os.path.basename(r.get("Target")) for r in rels}

    # body sectPr (last sectPr directly under body)
    body = root.find(qn("body"))
    body_sect = body.find(qn("sectPr"))

    img6_para = None
    caption6_para = None
    for d in root.iter():
        if d.tag not in (WPn + "inline", WPn + "anchor"):
            continue
        blip = d.find(".//" + An + "blip")
        media = relmap.get(blip.get(Rn + "embed")) if blip is not None else None
        if media not in WIDTH_CM:
            continue
        cx = int(WIDTH_CM[media] * EMU)
        cy = int(cx / ratios[media])
        ext = d.find(WPn + "extent")
        ext.set("cx", str(cx))
        ext.set("cy", str(cy))
        for aext in d.iter(An + "ext"):
            aext.set("cx", str(cx))
            aext.set("cy", str(cy))
        # clear inherited first-line indent and center the figure paragraph
        fp = ancestor_para(d)
        if fp is not None:
            pPr = fp.find(qn("pPr"))
            if pPr is None:
                pPr = etree.Element(qn("pPr"))
                fp.insert(0, pPr)
            old_ind = pPr.find(qn("ind"))
            if old_ind is not None:
                pPr.remove(old_ind)
            ind = etree.SubElement(pPr, qn("ind"))
            ind.set(qn("left"), "0")
            ind.set(qn("right"), "0")
            ind.set(qn("firstLine"), "0")
            ind.set(qn("firstLineChars"), "0")
            old_jc = pPr.find(qn("jc"))
            if old_jc is not None:
                pPr.remove(old_jc)
            jc = etree.SubElement(pPr, qn("jc"))
            jc.set(qn("val"), "center")
        if media == "image6.png":
            img6_para = ancestor_para(d)

    # locate 图6 caption paragraph
    paras = list(root.iter(qn("p")))
    if img6_para is not None:
        idx = paras.index(img6_para)
        for p in paras[idx:idx + 3]:
            if para_text(p).strip().startswith("图6"):
                caption6_para = p
                break

    # insert continuous section breaks: [before 图6] end 2-col; [caption6] end 1-col
    if img6_para is not None and caption6_para is not None and body_sect is not None:
        para_before = img6_para.getprevious()
        while para_before is not None and para_before.tag != qn("p"):
            para_before = para_before.getprevious()

        def make_sect(num_cols, space):
            s = copy.deepcopy(body_sect)
            for child in list(s):
                s.remove(child)
            # copy page setup from body
            for tag in ("pgSz", "pgMar", "headerReference", "footerReference"):
                src = body_sect.find(qn(tag))
                if src is not None:
                    s.append(copy.deepcopy(src))
            typ = etree.SubElement(s, qn("type"))
            typ.set(qn("val"), "continuous")
            cols = etree.SubElement(s, qn("cols"))
            cols.set(qn("num"), str(num_cols))
            cols.set(qn("space"), str(space))
            return s

        def attach_sect(p, sect):
            pPr = p.find(qn("pPr"))
            if pPr is None:
                pPr = etree.Element(qn("pPr"))
                p.insert(0, pPr)
            old = pPr.find(qn("sectPr"))
            if old is not None:
                pPr.remove(old)
            pPr.append(sect)

        if para_before is not None:
            attach_sect(para_before, make_sect(2, 425))
        # caption ends the single-column (spanning) section
        attach_sect(caption6_para, make_sect(1, 425))

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("image6 para found:", img6_para is not None, "caption6:", caption6_para is not None)
    print("saved")


if __name__ == "__main__":
    main()
