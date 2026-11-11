# -*- coding: utf-8 -*-
"""Embed 图1 as SVG (vector) with PNG fallback, and fix its display size."""
import io
import os
import shutil
import zipfile

from lxml import etree
from PIL import Image

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_svg_fig1.docx"
PNG = r"F:\project\PyProject\GNN\StructSearch_v2\paper\figures\final_fig1.png"
SVG = r"F:\project\PyProject\GNN\StructSearch_v2\paper\figures\final_fig1.svg"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
ASVG = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
SVG_EXT_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
EMU = 360000
WPn, An, Rn = "{%s}" % WP, "{%s}" % A, "{%s}" % R


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}

    with open(PNG, "rb") as f:
        png = f.read()
    with open(SVG, "rb") as f:
        svg = f.read()
    parts["word/media/image1.png"] = png
    parts["word/media/image1.svg"] = svg
    im = Image.open(io.BytesIO(png))
    ratio = im.width / im.height

    # content types: default svg
    cts = etree.fromstring(parts["[Content_Types].xml"])
    if not any(d.get("Extension") == "svg" for d in cts.iter("{%s}Default" % CT)):
        d = etree.SubElement(cts, "{%s}Default" % CT)
        d.set("Extension", "svg")
        d.set("ContentType", "image/svg+xml")
    parts["[Content_Types].xml"] = etree.tostring(cts, xml_declaration=True, encoding="UTF-8", standalone=True)

    # relationship for svg
    rels = etree.fromstring(parts["word/_rels/document.xml.rels"])
    png_rid = None
    for r in rels:
        if r.get("Target") == "media/image1.png":
            png_rid = r.get("Id")
    rid_svg = "rIdSvgFig1"
    while any(r.get("Id") == rid_svg for r in rels.iter("{%s}Relationship" % REL)):
        rid_svg += "x"
    nr = etree.SubElement(rels, "{%s}Relationship" % REL)
    nr.set("Id", rid_svg)
    nr.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    nr.set("Target", "media/image1.svg")
    parts["word/_rels/document.xml.rels"] = etree.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone=True)

    # document.xml: add svgBlip ext to image1's blip + fix extent
    root = etree.fromstring(parts["word/document.xml"])
    for d in root.iter():
        if d.tag not in (WPn + "inline", WPn + "anchor"):
            continue
        blip = d.find(".//" + An + "blip")
        if blip is None:
            continue
        if blip.get(Rn + "embed") != png_rid:
            continue
        # svg ext
        extLst = blip.find(An + "extLst")
        if extLst is None:
            extLst = etree.SubElement(blip, An + "extLst")
        ext = etree.SubElement(extLst, An + "ext")
        ext.set("uri", SVG_EXT_URI)
        svgblip = etree.SubElement(ext, "{%s}svgBlip" % ASVG)
        svgblip.set(Rn + "embed", rid_svg)
        # extent
        cx = int(7.8 * EMU); cy = int(cx / ratio)
        e = d.find(WPn + "extent"); e.set("cx", str(cx)); e.set("cy", str(cy))
        for aext in d.iter(An + "ext"):
            aext.set("cx", str(cx)); aext.set("cy", str(cy))
        break

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("fig1 embedded as SVG (vector) + PNG fallback; display 7.8 x %.2f cm" % (7.8 / ratio))


if __name__ == "__main__":
    main()
