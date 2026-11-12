# -*- coding: utf-8 -*-
"""Repair: reorder NDCG intro (preserve comment anchors) and re-anchor the lost Liao comment."""
import copy
import os
import shutil
import zipfile

from lxml import etree

DOCX = (r"F:\project\PyProject\GNN\StructSearch_v2\papers"
        r"\面向建筑结构案例的两阶段相似性检索方法_论文n1_261112_引用精简.docx")
BAK = r"C:\Users\246\AppData\Local\Temp\opencode\before_repair.docx"
BEFORE_MOVE = r"C:\Users\246\AppData\Local\Temp\opencode\before_move_metrics.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"

OLD_KEY = "本文采用信息检索评价指标进行检索排序性能分析"
NEW = ("本文采用信息检索评价指标进行检索排序性能分析。对面向工程应用的结构案例检索而言，检索结果的工程参考"
       "价值最为直观：MeanScore@K定义为前K个检索结果的平均专家评分，直接反映工程师对检索结果的可用性判断；"
       "Recall@K度量前K个结果召回的相关案例（评分≥3）比例，反映方法对高相关案例的覆盖能力。归一化折损累计"
       "增益（normalized discounted cumulative gain，NDCG）是一种综合考虑相关案例在排序中位置与专家评分等级的"
       "排序质量指标，本文按照Järvelin和Kekäläinen提出的方法计算[10]，以评估检索结果的整体排序质量：")
P31_KEY = "案例检索通常将案例表示为特征向量"


def qn(tag):
    return "{%s}%s" % (W, tag)


def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("t")))


def main():
    shutil.copyfile(DOCX, BAK)
    with zipfile.ZipFile(DOCX, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])
    comments = etree.fromstring(parts["word/comments.xml"])

    # ---- 1) NDCG paragraph: replace text, keep comment markers ----
    p_ndcg = [p for p in root.iter(qn("p")) if para_text(p).startswith(OLD_KEY)]
    if len(p_ndcg) != 1:
        raise RuntimeError("NDCG para matched %d" % len(p_ndcg))
    p_ndcg = p_ndcg[0]
    ts = [t for t in p_ndcg.iter(qn("t"))]
    if not ts:
        raise RuntimeError("no text run in NDCG para")
    ts[0].text = NEW
    ts[0].set("{%s}space" % XML, "preserve")
    for t in ts[1:]:
        t.text = ""

    # ---- 2) re-anchor Liao comment as id=7 in 3.1 first paragraph ----
    existing_ids = {c.get(qn("id")) for c in comments.findall(qn("comment"))}
    new_id = str(max(int(i) for i in existing_ids) + 1)
    p31 = [p for p in root.iter(qn("p")) if para_text(p).startswith(P31_KEY)]
    if len(p31) != 1:
        raise RuntimeError("3.1 first para matched %d" % len(p31))
    p31 = p31[0]

    # clone a comment-reference run for styling
    ref_run = None
    for r in p_ndcg.findall(qn("r")):
        if r.find(qn("commentReference")) is not None:
            ref_run = copy.deepcopy(r)
            break
    if ref_run is None:
        raise RuntimeError("no comment reference run to clone")

    start = etree.Element(qn("commentRangeStart"))
    start.set(qn("id"), new_id)
    end = etree.Element(qn("commentRangeEnd"))
    end.set(qn("id"), new_id)
    ref_run.find(qn("commentReference")).set(qn("id"), new_id)

    # insert start right after pPr, append end + reference at paragraph end
    pPr = p31.find(qn("pPr"))
    idx = 0 if pPr is None else list(p31).index(pPr) + 1
    p31.insert(idx, start)
    p31.append(end)
    p31.append(ref_run)

    # ---- 3) append cloned Liao comment to comments.xml ----
    zbm = zipfile.ZipFile(BEFORE_MOVE)
    cbm = etree.fromstring(zbm.read("word/comments.xml"))
    cm4 = [c for c in cbm.findall(qn("comment")) if c.get(qn("id")) == "4"][0]
    new_cm = copy.deepcopy(cm4)
    new_cm.set(qn("id"), new_id)
    comments.append(new_cm)

    parts["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    parts["word/comments.xml"] = etree.tostring(comments, xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = DOCX + ".tmp2"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    os.replace(tmp, DOCX)
    print("repair done: NDCG reordered, Liao comment re-anchored id=%s" % new_id)


if __name__ == "__main__":
    main()
