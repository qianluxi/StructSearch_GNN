# -*- coding: utf-8 -*-
"""Re-embed the final (larger-font) figures and update display sizes by new ratios."""
import io
import os
import shutil
import zipfile

from lxml import etree
from PIL import Image

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_reembed_fonts.docx"
FIGDIR = r"F:\project\PyProject\GNN\StructSearch_v2\paper\figures"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
EMU = 360000
WPn, An, Rn = "{%s}" % WP, "{%s}" % A, "{%s}" % R

# media -> (source file, target width cm)
MAP = {
    "image1.png": ("final_fig1.png", 7.8),
    "image2.png": ("final_fig2.png", 7.8),
    "image3.png": ("final_fig3.png", 7.8),
    "image4.png": ("final_fig4.png", 7.8),
    "image5.png": ("final_fig5.png", 7.8),
    "image6.png": ("final_fig6.png", 16.5),
}


def qn(tag):
    return "{%s}%s" % (W, tag)


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}

    ratios = {}
    for media, (src, _w) in MAP.items():
        with open(os.path.join(FIGDIR, src), "rb") as f:
            data = f.read()
        parts["word/media/" + media] = data
        im = Image.open(io.BytesIO(data))
        ratios[media] = im.width / im.height

    root = etree.fromstring(parts["word/document.xml"])
    rels = etree.fromstring(parts["word/_rels/document.xml.rels"])
    relmap = {r.get("Id"): os.path.basename(r.get("Target")) for r in rels}

    for d in root.iter():
        if d.tag not in (WPn + "inline", WPn + "anchor"):
            continue
        blip = d.find(".//" + An + "blip")
        media = relmap.get(blip.get(Rn + "embed")) if blip is not None else None
        if media not in MAP:
            continue
        wcm = MAP[media][1]
        cx = int(wcm * EMU)
        cy = int(cx / ratios[media])
        ext = d.find(WPn + "extent")
        ext.set("cx", str(cx)); ext.set("cy", str(cy))
        for aext in d.iter(An + "ext"):
            aext.set("cx", str(cx)); aext.set("cy", str(cy))

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    for media in MAP:
        print(media, "ratio %.3f" % ratios[media])
    print("saved")


if __name__ == "__main__":
    main()
