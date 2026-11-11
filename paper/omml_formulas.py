# -*- coding: utf-8 -*-
"""Replace plain-text equations with native Word equations (OMML)."""
import os
import shutil
import zipfile

import latex2mathml.converter as l2m
import mathml2omml
from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261109_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_omml.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML = "http://www.w3.org/XML/1998/namespace"
Wn, Mn = "{%s}" % W, "{%s}" % M

DISPLAY = [
    ("x = [", "1",
     r"\begin{split} x = [p_1, p_2, p_3, q_1, q_2, q_3, \\ \ln(1+S), \mathrm{CV}, H, \mu_{nn}, \sigma_{nn}]^{\mathrm{T}} \in \mathbb{R}^{11} \end{split}"),
    ("zᵢ =", "2",
     r"z_i = \frac{x_i - \mu_i}{\sigma_i},\quad i = 1, 2, \ldots, 11"),
    ("C(q) =", "3",
     r"C(q) = \{x_j \mid \mathrm{type}(x_j) = \mathrm{type}(q),\; x_j \neq q\}"),
    ("d(q,", "4",
     r"d(q, x_j) = \sum_{i=1}^{11} |z_{q,i} - z_{j,i}|"),
    ("R_K(q)", "5",
     r"R_K(q) = \mathrm{TopK}_{x_j \in C(q)}[-d(q, x_j)]"),
    ("NDCG@K", "6",
     r"\mathrm{NDCG}@K = \frac{\sum_{i=1}^{K} \frac{2^{\mathrm{rel}_i} - 1}{\log_2(1+i)}}{\mathrm{IDCG}@K}"),
]
SOFTMAX = ("wⱼ = exp(sⱼ) / Σₖ₌₁³ exp(sₖ)",
           r"w_j = \frac{\exp(s_j)}{\sum_{k=1}^{3}\exp(s_k)}")


def qn(tag):
    return "{%s}%s" % (W, tag)


def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("t")))


def omml(latex):
    ml = l2m.convert(latex)
    frag = mathml2omml.convert(ml)
    wrapper = ('<root xmlns:m="%s">%s</root>' % (M, frag)).encode("utf-8")
    r = etree.fromstring(wrapper)
    return r[0]


def rpr_from(p):
    for run in p.findall(qn("r")):
        if run.find(qn("rPr")) is not None:
            return run.find(qn("rPr"))
    return None


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])
    body_paras = list(root.iter(qn("p")))

    # display equations
    for anchor, num, latex in DISPLAY:
        hits = [p for p in body_paras if para_text(p).strip().startswith(anchor)]
        if len(hits) != 1:
            raise RuntimeError("anchor %r matched %d" % (anchor, len(hits)))
        p = hits[0]
        rpr = rpr_from(p)
        for child in list(p):
            if child.tag == qn("pPr"):
                continue
            p.remove(child)
        pPr = p.find(qn("pPr"))
        if pPr is None:
            pPr = etree.Element(qn("pPr"))
            p.insert(0, pPr)
        old = pPr.find(qn("jc"))
        if old is not None:
            pPr.remove(old)
        jc = etree.SubElement(pPr, qn("jc")); jc.set(qn("val"), "center")

        p.append(omml(latex))
        run = etree.SubElement(p, qn("r"))
        if rpr is not None:
            run.append(etree.fromstring(etree.tostring(rpr)))
        t = etree.SubElement(run, qn("t"))
        t.set("{%s}space" % XML, "preserve")
        t.text = "　　(%s)" % num

    # inline softmax
    hits = [p for p in body_paras if SOFTMAX[0] in para_text(p)]
    if len(hits) != 1:
        raise RuntimeError("softmax paragraph matched %d" % len(hits))
    p = hits[0]
    text = para_text(p)
    before, after = text.split(SOFTMAX[0], 1)
    rpr = rpr_from(p)
    for child in list(p):
        if child.tag == qn("pPr"):
            continue
        p.remove(child)

    def text_run(txt):
        r = etree.SubElement(p, qn("r"))
        if rpr is not None:
            r.append(etree.fromstring(etree.tostring(rpr)))
        t = etree.SubElement(r, qn("t"))
        t.set("{%s}space" % XML, "preserve")
        t.text = txt

    text_run(before)
    p.append(omml(SOFTMAX[1]))
    text_run(after)

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("inserted OMML: 6 display + 1 inline")


if __name__ == "__main__":
    main()
