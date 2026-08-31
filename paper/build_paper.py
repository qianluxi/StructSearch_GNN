"""按《建筑结构》期刊格式（参照 部分框支剪力墙...210108.docx 模板）生成论文。

修订版（依据专家审稿意见）：
- 新增“1 相关研究”章节并明确研究空白；
- 贡献重组为 3 条（描述结构 / 约束检索 / 正确评价）；
- 消融实验更名为“结构体系判别特征消融实验”并说明口径；
- 全文统计表述修订（去“显著”、留一口径、置信度声明、n=8 诚实表述）；
- 新增项目身份置换鲁棒性验证表与典型案例分析（图6）；
- 参考文献扩充至 19 篇真实文献。
"""

import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATE = os.path.join(ROOT, "部分框支剪力墙结构抗震性能目标选择的探讨_210108.docx")
OUT = os.path.join(ROOT, "面向建筑结构案例的两阶段相似性检索方法_论文.docx")
FIG = os.path.join(HERE, "figures")


def clear_indent(par):
    pPr = par._p.find(qn("w:pPr"))
    if pPr is not None:
        ind = pPr.find(qn("w:ind"))
        if ind is not None:
            pPr.remove(ind)


def set_run_font(run, size=None, east_asia=None):
    if size is not None:
        run.font.size = Pt(size)
    if east_asia:
        rPr = run._element.get_or_add_rPr()
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = rPr.makeelement(qn("w:rFonts"), {})
            rPr.append(rFonts)
        rFonts.set(qn("w:eastAsia"), east_asia)


def add_three_line_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = tblPr.makeelement(qn("w:tblBorders"), {})
    for edge, sz in (("top", "12"), ("bottom", "12")):
        el = borders.makeelement(qn(f"w:{edge}"), {
            qn("w:val"): "single", qn("w:sz"): sz,
            qn("w:space"): "0", qn("w:color"): "000000"})
        borders.append(el)
    for edge in ("left", "right", "insideH", "insideV"):
        el = borders.makeelement(qn(f"w:{edge}"), {
            qn("w:val"): "none", qn("w:sz"): "0",
            qn("w:space"): "0", qn("w:color"): "auto"})
        borders.append(el)
    tblPr.append(borders)
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcBorders = tcPr.makeelement(qn("w:tcBorders"), {})
        bottom = tcBorders.makeelement(qn("w:bottom"), {
            qn("w:val"): "single", qn("w:sz"): "6",
            qn("w:space"): "0", qn("w:color"): "000000"})
        tcBorders.append(bottom)
        tcPr.append(tcBorders)


def fill_cell(cell, text, bold=False):
    par = cell.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    clear_indent(par)
    run = par.add_run(text)
    run.bold = bold
    set_run_font(run, size=9, east_asia="宋体")


def main():
    doc = Document(TEMPLATE)
    body = doc.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)

    def title(text):
        p = doc.add_paragraph(style="Title")
        clear_indent(p)
        p.add_run(text)

    def centered(text, size=None, bold=False, italic=False):
        p = doc.add_paragraph(style="Normal")
        clear_indent(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.bold = bold
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        return p

    def normal(text):
        p = doc.add_paragraph(text, style="Normal")
        return p

    def label_para(label, text):
        p = doc.add_paragraph(style="Normal")
        clear_indent(p)
        r = p.add_run(label)
        r.bold = True
        p.add_run(text)

    def h1(text):
        doc.add_paragraph(text, style="Heading 1")

    def h2(text):
        doc.add_paragraph(text, style="Heading 2")

    def caption(text):
        doc.add_paragraph(text, style="图名")

    def equation(text):
        p = doc.add_paragraph(style="Normal")
        clear_indent(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.italic = True
        return p

    def figure(path, cap, width_cm):
        p = doc.add_paragraph()
        clear_indent(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=Cm(width_cm))
        caption(cap)

    def table(cap, header, rows, bold_rows=(), note=None):
        caption(cap)
        t = doc.add_table(rows=len(rows) + 1, cols=len(header))
        t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_three_line_borders(t)
        for j, htext in enumerate(header):
            fill_cell(t.rows[0].cells[j], htext, bold=True)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                fill_cell(t.rows[i + 1].cells[j], val, bold=(i in bold_rows))
        if note:
            caption(note)

    def ref(text):
        doc.add_paragraph(text, style="endnote text")

    # ================= 标题与作者 =================
    title("面向建筑结构案例的两阶段相似性检索方法")
    centered("A Two-stage Structural Similarity Retrieval Method for Building Structure Cases",
             size=12, italic=True)
    centered("钱路曦")
    centered("（浙江绿城建筑设计有限公司，杭州 310007）")

    # ================= 摘要 =================
    label_para("[摘　要]　",
               "建筑结构设计过程中，设计人员常需从历史图纸中检索与当前工程结构相似的案例，"
               "作为方案比选与类比分析的参考，但传统图纸管理系统仅能依据项目名称、图纸编号等"
               "元数据检索，无法表达结构相似性，且结构图纸之间的“结构相似”长期缺少明确的"
               "量化定义与评价方法。为此，本文提出一种结构体系约束的两阶段结构相似性检索方法。"
               "该方法对DXF结构图纸进行解析，将构件划分为柱、短肢剪力墙与剪力墙，并按照"
               "“结构体系—构件权重—尺寸规模—尺寸离散—空间布局”的信息层级构建11维结构统计"
               "描述子；第一阶段采用留一检索式余弦近邻法识别图纸的结构体系类型（框架、框剪、"
               "剪力墙），第二阶段在同类型候选集内以标准化描述子与曼哈顿距离进行Top-K相似案例"
               "检索。同时构建了包含15个查询、171对0～5分盲评标注的专家结构相似性评价基准，"
               "提出并验证了“检索项目无关、评估项目感知”原则。实验结果表明：库内留一检索式结构"
               "体系识别Top-1准确率为87.7%；标准化统计描述子配合曼哈顿距离的检索性能最优"
               "（NDCG@3=0.746，Recall@3=35.6%，MeanScore@3=1.111）；在当前案例库与评价设置"
               "下，构件类型占比是最具判别力的特征组；在本研究专家样本的跨项目子集中，统计描述子"
               "（NDCG@5=0.501）明显高于以分类为目标训练的图神经网络表征（0.273）。研究成果"
               "表明，具有明确工程语义的结构描述子能够有效支持细粒度结构案例检索，可为结构设计"
               "中的历史案例复用提供量化工具。")
    label_para("[关键词]　", "建筑结构；案例检索；结构相似性；结构体系识别；结构统计描述子；专家评价")
    label_para("[Abstract]　",
               "In structural design, engineers often need to retrieve historically similar cases for "
               "scheme comparison and analogical analysis, while conventional drawing management "
               "systems retrieve only by metadata such as project names and drawing numbers and cannot "
               "express structural similarity, and the notion of structural similarity between drawings "
               "has long lacked a quantitative definition and evaluation method. This paper proposes a "
               "two-stage structural similarity retrieval framework constrained by structural system "
               "type. DXF structural drawings are parsed, members are classified into columns, "
               "short-pier shear walls and shear walls, and an 11-dimensional structural statistical "
               "descriptor is built following an information hierarchy of structural system, member "
               "weights, size scale, size dispersion and spatial layout. In Stage 1, the structural "
               "system type (frame, frame-shear wall or shear wall) of the query drawing is identified "
               "by leave-one-out cosine nearest-neighbor classification; in Stage 2, Top-K similar case "
               "retrieval is performed within the same-type candidate set using the standardized "
               "descriptor with Manhattan distance. An expert structural similarity benchmark is "
               "constructed, containing 15 queries and 171 blind annotations on a 0-5 scale, following "
               "a project-blind retrieval and project-aware evaluation protocol. Experimental results "
               "show that leave-one-out type recognition reaches a Top-1 accuracy of 87.7%; the "
               "standardized descriptor with Manhattan distance achieves the best retrieval performance "
               "(NDCG@3=0.746, Recall@3=35.6%, MeanScore@3=1.111); under the current case library and "
               "evaluation setting, the component type ratio is the most discriminative feature group; "
               "on the cross-project subset of our expert sample, the statistical descriptor "
               "(NDCG@5=0.501) clearly outperforms the graph neural network representation trained "
               "with a classification objective (0.273). The results indicate that descriptors with "
               "explicit engineering semantics can effectively support fine-grained structural case "
               "retrieval.")
    label_para("[Keywords]　",
               "building structure; case retrieval; structural similarity; structural system "
               "recognition; structural statistical descriptor; expert evaluation")

    # ================= 0 引言 =================
    h1("0 引言")
    normal("建筑结构设计过程中，设计人员经常需要参考历史工程案例：方案比选时，需要寻找结构体系、"
           "构件组成与几何尺度相近的已建成方案作为参照；超限审查与专项论证中的类比分析，同样依赖"
           "相似历史案例的经验。随着计算机辅助设计的普及，设计企业积累了大量以DXF格式存储的数字化"
           "结构图纸档案，构成了宝贵的设计知识库。然而，现有图纸管理系统主要依赖项目名称、图纸编号、"
           "专业工种等元数据进行检索，无法回答“当前图纸与哪个历史案例在结构上最相似”这一问题，"
           "结构相似案例的获取长期依赖人工记忆与经验。基于案例的推理（case-based reasoning，CBR）"
           "研究表明，复用历史案例能够显著提升设计效率[1-3]，但其前提是建立可靠的案例表示、相似性"
           "度量与检索机制——而对结构图纸而言，“结构相似”本身尚缺少明确的量化定义。")
    normal("与通用图像、文本检索相比，建筑结构图纸的相似性检索具有三个特殊挑战。其一，结构图纸的"
           "对象是二维工程几何，其承载的是构件类型组成、构件尺度与空间布置等信息，简单的像素级图像"
           "相似度不能反映结构相似性；其二，结构体系是建筑结构的根本属性，建筑抗震设计相关标准对"
           "框架、框剪、剪力墙等体系的适用高度、构件构成与抗震措施有严格规定[4-5]，不同体系的结构"
           "即使某些几何统计量接近，也不应被视为高度相似，若在全空间直接比较容易产生误匹配，因此"
           "结构相似性检索需要引入结构体系这一工程先验作为约束；其三，项目归属不能等同于结构相似——"
           "同一项目可能包含体系、布置、高度各不相同的多个楼栋，不同项目之间也可能采用非常接近的"
           "结构方案，若直接以“同项目”作为相似性的代理标签，将给检索评价引入系统性偏差，必须建立"
           "专家定义的结构相似性评价体系。")
    normal("据此，本文将研究问题定义为：如何从建筑结构图纸出发，定量地定义、计算并评价图纸之间的"
           "结构相似性，从而在历史案例库中自动检索出具有工程参考价值的相似案例。这是一个此前缺少"
           "明确量化方法的问题，本文以工程案例验证的方式对其进行探索。围绕该问题，本文的主要贡献"
           "包括三个方面：（1）建立面向建筑结构图纸的工程语义结构描述方法，即按“结构体系—构件权重—"
           "尺寸规模—尺寸离散—空间布局”信息层级组织的11维结构统计描述子，将结构相似性的比较建立在"
           "具有明确工程含义的描述空间之上；（2）提出结构体系约束的两阶段结构相似性检索框架，以"
           "“先判型、后类内检索”的机制将结构体系转化为相似性候选空间的约束条件，使相似性比较只在"
           "同一结构体系内进行；（3）建立以专家评分为核心、项目名称与检索解耦的结构相似性评价方法，"
           "构建15个查询/171对盲评标注的专家基准，提出并实证验证“检索项目无关、评估项目感知”原则，"
           "并以此系统比较工程语义统计描述与以分类为目标训练的深度图表征在该任务上的表现。")

    # ================= 1 相关研究 =================
    h1("1 相关研究")
    h2("1.1建筑结构案例复用与基于案例的推理")
    normal("基于案例的推理源于认知科学对类比推理的研究，其核心思想是：新问题可以参照与其相似的"
           "历史案例求解[1]。Aamodt与Plaza将CBR归纳为检索、复用、修正与保存四个环节，并指出案例"
           "表示与相似性度量是整个方法的基础[2]；Watson进一步讨论了CBR在企业系统中的工程化实现[3]。"
           "在工程设计领域，结构方案的类比复用具有天然的工程价值：结构体系选型、构件布置与尺度确定"
           "都高度依赖既有经验。然而，结构案例的表示长期停留在项目名称、层数、高度等元数据层面，"
           "“结构相似”缺少面向图纸内容的量化描述，案例检索因此难以自动化。")
    h2("1.2工程图纸与BIM模型的相似性检索")
    normal("工程图纸与建筑信息模型（BIM）的数字化为案例检索提供了数据基础[6]。在通用层面，基于"
           "内容的图像检索（CBIR）经过数十年发展，已形成颜色、纹理、形状等多层次视觉特征体系[7]；"
           "在形状层面，基于轮廓的形状描述子被广泛用于非刚体形状匹配[8]。但这些方法面向的是视觉"
           "外观相似：结构平面图即使视觉轮廓接近，若构件类型组成与结构体系不同，在工程意义上也不"
           "相似；反之，绘制风格不同的图纸也可能承载非常接近的结构方案。直接套用图像或形状检索方法，"
           "无法表达结构工程所关心的语义相似。")
    h2("1.3图表示学习及其在结构检索中的应用前景")
    normal("建筑结构平面天然具有图结构：构件为节点，空间邻接或传力关系为边。图神经网络（graph "
           "neural network，GNN）通过消息传递聚合邻域信息，能够学习图级表征[10]，代表性方法包括"
           "图卷积网络（GCN）[11]、GraphSAGE[12]、图注意力网络（GAT）[13]与GIN[14]。在相似性学习"
           "方面，度量学习提供了以“相似样本距离近、不相似样本距离远”为目标的训练范式[15]，并在"
           "人脸识别等任务中得到验证[16-17]。因此，图表示学习具备用于结构平面检索的潜力，但其表征"
           "质量高度依赖监督信号的设计：若训练目标与检索任务不一致，学到的表征未必支持细粒度"
           "相似性排序。")
    h2("1.4研究空白")
    normal("综上，现有方法或面向图纸的几何与视觉外观[7-8]，或面向项目元数据与模型信息[6]，或提供"
           "通用的深度图表征工具[10-14]；而针对建筑结构体系及其工程语义构件组成的结构相似性检索"
           "研究仍较少，且已有评价方式多以“同项目”作为相似性的代理，缺少专家定义的结构相似性真值。"
           "本文正是围绕这一空白展开：以工程语义描述结构、以结构体系约束检索、以专家评分评价检索。")

    # ================= 2 案例库与结构描述 =================
    h1("2 结构案例库与结构描述")
    h2("2.1案例库构成")
    normal("本文案例库包含106张DXF格式的结构平面图纸，来自20个真实工程项目，涵盖建筑抗震设计相关"
           "标准定义的三种基本结构体系[4]：剪力墙结构42张、框剪结构38张、框架结构26张，图纸覆盖"
           "地下室、标准层、裙房等不同功能楼层。检索在库内进行，查询时排除查询图纸自身。案例库与"
           "专家基准的基本统计见表1。")
    table("表1 案例库与专家基准数据统计",
          ["项目", "数值"],
          [["图纸总数", "106"],
           ["剪力墙结构", "42"],
           ["框剪结构", "38"],
           ["框架结构", "26"],
           ["覆盖工程项目数", "20"],
           ["专家基准查询数", "15（三种类型各5）"],
           ["专家标注对数", "171"],
           ["专家评分范围", "0～5分"]])
    h2("2.2DXF结构解析与构件分类")
    normal("本文针对DXF格式的结构平面图建立解析流程：首先提取LWPOLYLINE与POLYLINE实体并转换为闭合"
           "多边形（首尾间隙小于1.0单位时自动闭合），对INSERT块引用实体按最大4层深度递归展开，以覆盖"
           "以块方式组织的构件图；随后以最小外接矩形对每个多边形进行主轴对齐，并保留构件质心坐标，"
           "使图纸描述具有平移与旋转不变性。")
    normal("依据几何特征将每个构件划分为柱、短肢剪力墙、剪力墙三类。记构件长肢长度与厚度之比为"
           "长细比r，分类规则为：当r≥8，或壁厚t≥310mm且r＞4时，判为剪力墙；当r≥4且不满足剪力墙条件"
           "时，判为短肢剪力墙；其余判为柱。每个构件记录质心坐标、面积、跨度、长肢长度、壁厚与长细比"
           "等属性，构成特征构建所需的构件集合。该分类规则与结构工程中柱、短肢剪力墙与剪力墙的判别"
           "习惯一致，使后续特征具有明确的工程含义。")
    h2("2.3结构统计描述子的设计与定义")
    normal("结构相似性的量化，首先需要回答结构工程师判断两个方案是否相似时所关注的信息。本文将这些"
           "关注点组织为六个信息层级：结构体系——这是什么类型的结构；构件权重——各类构件在结构中"
           "占多大比重；尺寸规模——结构的整体体量；尺寸离散——构件尺寸是否集中；构件均匀性——构件"
           "面积分布是否均匀；空间布局——构件布置是否疏密均匀。六个层级与具体特征的对应关系见表2。"
           "在这一信息层级下，11维描述子不是若干统计量的简单堆叠，而是从结构工程认知出发构造的结构"
           "描述空间。")
    table("表2 结构统计描述子的信息层级与特征设计",
          ["信息层级", "工程问题", "对应特征", "维数"],
          [["结构体系", "这是什么类型的结构", "构件类型占比p", "3"],
           ["构件权重", "各类构件贡献多大比重", "构件面积占比q", "3"],
           ["尺寸规模", "结构整体体量多大", "总面积对数ln(1+S)", "1"],
           ["尺寸离散", "构件尺寸是否集中", "面积变异系数CV", "1"],
           ["构件均匀性", "构件面积分布是否均匀", "面积熵H", "1"],
           ["空间布局", "构件布置疏密是否均匀", "最近邻距离统计μnn、σnn", "2"]])
    normal("对包含N个构件的结构平面，上述信息层级具体化为11维全局结构统计描述子：（1）构件类型占比"
           "（3维）：各类构件数量与构件总数之比；（2）构件面积占比（3维）：各类构件面积之和与全部构件"
           "总面积之比；（3）面积分布（3维）：总面积对数刻画结构规模，面积变异系数刻画构件尺寸的离散"
           "程度，面积熵刻画构件面积分布的均匀性；（4）空间统计（2维）：各构件最近邻质心距离的均值与"
           "标准差，刻画构件布置的疏密与均匀程度。完整描述子为：")
    equation("x = [p₁, p₂, p₃, q₁, q₂, q₃, ln(1+S), CV, H, μnn, σnn]ᵀ ∈ R¹¹　　(1)")
    normal("其中p₁～p₃与q₁～q₃分别为柱、短肢剪力墙、剪力墙的数量占比与面积占比；S为构件总面积；"
           "CV=σA/μA为面积变异系数；H=−Σαⱼlnαⱼ为面积熵，αⱼ为第j个构件面积占总面积的比例；μnn与σnn"
           "为最近邻质心距离的均值与标准差。各维度定义汇总于表3。描述子基于构件集合的统计量计算，"
           "对图纸的平移、旋转与绘制坐标系不敏感，且全程不使用图纸的项目名称等项目身份信息。")
    table("表3 11维结构统计描述子定义",
          ["维度", "符号", "工程含义"],
          [["1～3", "p₁, p₂, p₃", "柱、短肢剪力墙、剪力墙的数量占比"],
           ["4～6", "q₁, q₂, q₃", "柱、短肢剪力墙、剪力墙的面积占比"],
           ["7", "ln(1+S)", "构件总面积对数（结构规模）"],
           ["8", "CV", "面积变异系数（尺寸离散度）"],
           ["9", "H", "面积熵（尺寸分布均匀性）"],
           ["10～11", "μnn, σnn", "最近邻质心距离均值、标准差（布置疏密）"]])
    h2("2.4特征标准化")
    normal("上述11维特征量纲不同、取值范围不一，若直接在原始空间中度量距离，数值尺度大的维度将主导"
           "相似性计算。为此对每一维进行Z-score标准化：")
    equation("zᵢ = (xᵢ − μᵢ) / σᵢ，i = 1, 2, …, 11　　(2)")
    normal("其中μᵢ、σᵢ分别为案例库中第i维特征的样本均值与标准差（σᵢ接近0时取1）。标准化统计量仅来自"
           "案例库自身，不涉及任何标注信息。第5.2节实验将验证标准化对检索性能具有决定性影响。")

    # ================= 3 两阶段检索方法 =================
    h1("3 两阶段结构相似性检索方法")
    normal("两阶段结构相似性检索方法的总体框架如图1所示：对查询图纸先进行结构解析并构建结构统计"
           "描述子，随后依次经过“第一阶段结构体系识别”与“第二阶段类内相似检索”，输出Top-K相似案例。"
           "该框架遵循“识别先行、检索在后”的原则，将结构体系这一工程先验转化为检索过程的硬约束。")
    figure(os.path.join(FIG, "fig1_pipeline.png"), "图1 两阶段结构相似性检索框架", 11.5)
    h2("3.1第一阶段：结构体系识别")
    normal("结构体系识别采用“以检索代分类”的策略，不依赖任何训练模型：以库内其余图纸的标准化描述子"
           "为参照样本，计算查询描述子与全部参照样本的余弦相似度，取相似度最高的3个样本，以其结构体系"
           "类型多数投票确定预测类型[9]。需要说明两点：其一，本文报告的识别准确率采用库内留一检索式"
           "口径，即每张图纸在查询时以其余105张图纸为参照，不涉及独立测试集；其二，对3个近邻相似度"
           "作softmax归一化、以预测类型的累计概率作为“近邻投票置信度”，该置信度是相似度归一化的相对"
           "指标，不是统计意义上的校准分类概率，仅用于结果排序与人工复核参考。在106张图纸的库内留一"
           "口径下，该方法的Top-1识别准确率为87.7%，表明统计描述子对结构体系已具备较强的判别力。")
    h2("3.2第二阶段：类内相似检索")
    normal("确定查询图纸的结构类型后，仅在同类型候选集内进行检索。设查询q的同类型候选集为：")
    equation("C(q) = { xⱼ | type(xⱼ) = type(q)，xⱼ ≠ q }　　(3)")
    normal("本文对曼哈顿距离、欧氏距离与余弦相似度三种度量方式进行比较。以曼哈顿距离为例，查询与"
           "候选之间的距离为：")
    equation("d(q, xⱼ) = Σᵢ₌₁¹¹ | zq,ᵢ − zⱼ,ᵢ |　　(4)")
    normal("最终检索函数定义为候选集内按距离升序取前K个：")
    equation("R_K(q) = TopK_{xⱼ ∈ C(q)} [ −d(q, xⱼ) ]　　(5)")
    normal("该策略保证结构相似性只在同一结构体系内比较，避免了不同体系之间因统计量巧合接近而产生的"
           "误匹配，与结构工程中“先看体系、再看细部”的认知习惯一致。式(3)～(5)共同构成了从描述子到"
           "检索结果的完整数学链条：x → z → type(q) → C(q) → d(q, xⱼ) → R_K(q)。")
    h2("3.3深度图表征对照方法")
    normal("为考察深度表征学习方法在该任务上的表现，本文以图神经网络作为对照方法[10,12]：将结构平面"
           "构件集合建模为无向图，构件为节点，以质心K近邻（k=6）确定节点连接，节点间欧氏距离作为边"
           "属性；节点特征为构件类型独热编码、面积、跨度、壁厚与长细比等9维。图编码器采用3层GraphSAGE"
           "卷积[12]加全局均值池化与线性投影，输出128维图嵌入与3类分类logits，以结构类型分类的交叉熵"
           "损失训练（训练/验证按80/20划分，随机种子42），取验证准确率最优的检查点。GNN检索以128维"
           "嵌入的余弦相似度排序；Hybrid方法将128维GNN嵌入与标准化11维统计描述子拼接为139维向量后作"
           "余弦检索。需要强调，GNN在本文中仅作为深度对照方法，两阶段检索框架本身不依赖任何训练模型。")

    # ================= 4 专家评价基准 =================
    h1("4 结构相似性专家评价基准")
    h2("4.1项目归属不能作为结构相似性真值")
    normal("以往案例检索研究常以“同项目”作为案例相关性的近似判据。为检验这一代理的可靠性，本文首先"
           "邀请结构工程师对76对案例进行0～3分的相似性评分（0=不相似，3=高度相似），统计结果见表4。")
    table("表4 同项目与跨项目案例对专家评分对比（0～3分制，76对）",
          ["对类型", "对数", "平均分", "≥2分比例"],
          [["同项目", "12", "1.58", "50%"],
           ["跨项目", "64", "0.55", "11%"]])
    normal("结果显示，同项目案例对的平均相似度确实更高，项目归属具有一定先验价值；但同项目对中有一半"
           "被专家判为不相似（评分≤1），而跨项目对中仍有11%被判断为具有参考价值。这说明“同项目”只能"
           "作为弱标签，不能作为结构相似性的真值。因此本文后续评价以专家评分为唯一相关性依据，项目归属"
           "仅作为评估分析维度。")
    h2("4.2专家评分协议与基准构建")
    normal("基于上述认识，本文构建专家结构相似性评价基准。查询选取：从5个不同工程项目中选取15个查询，"
           "框架、框剪、剪力墙三类各5个；其中刻意包含所在项目仅1张图纸的查询，使该查询只能从跨项目候选"
           "中获得相似案例，避免“同项目自动存在正样本”的评价捷径。候选生成：对每个查询，将三条检索路线"
           "（统计描述子、GNN、Hybrid）Top-5结果的并集与2个困难负样本（同类型但最不相似）作为待评候选，"
           "共获得171对有效标注。")
    normal("评分协议方面，评分由一名具有结构设计经验的工程师完成：评分表仅包含查询与候选图纸，"
           "不显示候选来自哪条检索路线（对方法来源盲评），以避免评价偏向某一方法；图纸文件名中自然"
           "包含项目归属信息，这与本文“评估项目感知”的定位一致。评分采用0～5分等级制（表5）。评分"
           "分布为：0分109对、1分39对、2分12对、3分9对、4分2对、5分0对；按评分≥3为相关的口径，共有"
           "11对相关对。专家评分总体偏严，与工程上真正结构相似的案例稀少的实际情况相符，因此本文"
           "更强调方法间的相对比较而非指标绝对值。需要说明，当前标注由单一专家完成，尚未进行多专家"
           "一致性检验，该问题在第6.5节局限性部分进一步讨论。")
    table("表5 专家相似性评分等级",
          ["评分", "含义"],
          [["0", "基本不相似，无参考价值"],
           ["1", "弱相关"],
           ["2", "有一定相似性"],
           ["3", "有明确参考价值（相关阈值）"],
           ["4", "很相似，参考价值高"],
           ["5", "高度相似，可直接作为主要参考"]])
    h2("4.3评价指标")
    normal("本文采用信息检索的标准指标进行评价[18]。归一化折损累计增益（normalized discounted "
           "cumulative gain，NDCG）综合考虑相关案例在排序中的位置与专家评分等级：")
    equation("NDCG@K = Σᵢ₌₁ᴷ (2^relᵢ − 1) / log₂(1 + i) ÷ IDCG@K　　(6)")
    normal("其中relᵢ为排序第i位案例的专家评分，IDCG@K为理想排序下可达到的最大值。Recall@K度量前K个"
           "结果召回的相关案例比例；MeanScore@K为前K个结果的平均专家评分，直接反映检索结果的工程"
           "参考价值，是面向工程师视角的指标；相关判据为专家评分≥3。需要指出，由于NDCG采用分级增益，"
           "即使随机排序也能命中低分案例而获得不低的NDCG，因此本文以Recall与MeanScore作为主要参照，"
           "NDCG用于方法间相对排序。")
    h2("4.4项目身份置换鲁棒性验证")
    normal("在方法学原则上，本文严格执行“检索项目无关、评估项目感知”：图纸的项目名称不参与特征构建、"
           "类型识别与相似度排序的任何计算，仅在评估阶段用于同项目/跨项目子集拆分。为实证验证该原则，"
           "本文进行了项目名置换检验：对20个项目名进行两次确定性改名（随机种子42与7，分别改名19与20个"
           "项目名），重新运行全部检索流程。结果如表6所示，结构体系识别结果与三条检索路线的全部指标在"
           "置换前后完全一致，表明检索系统对项目身份置换不变，项目信息确实未参与检索计算。")
    table("表6 项目身份置换鲁棒性验证",
          ["指标", "原始项目名", "置换1（种子42）", "置换2（种子7）"],
          [["结构体系识别Top-1准确率", "87.7%", "87.7%", "87.7%"],
           ["统计路线 NDCG@3", "0.7214", "0.7214", "0.7214"],
           ["统计路线 Recall@3", "26.7%", "26.7%", "26.7%"],
           ["GNN路线 NDCG@3", "0.5947", "0.5947", "0.5947"],
           ["Hybrid路线 NDCG@3", "0.7135", "0.7135", "0.7135"],
           ["弱标签检索 MRR", "0.5800", "0.5800", "0.5800"]],
          note="注：统计路线此处为两阶段流水线默认度量（余弦相似度），与表7中“统计+余弦相似度”一致。")

    # ================= 5 实验结果与分析 =================
    h1("5 实验结果与分析")
    h2("5.1实验设置")
    normal("全部实验在CPU上完成且结果确定（随机种子42）；构图采用KNN，k=6；GNN训练采用80/20的训练/"
           "验证划分。评价数据为第4.2节的专家基准（15个查询/171对标注）。对比方法包括：Random（随机"
           "排序，性能参照）、未标准化的原始统计描述子、GNN、Hybrid，以及分别采用欧氏距离、余弦相似度"
           "与曼哈顿距离的标准化统计描述子。")
    h2("5.2检索方法对比")
    normal("各方法在专家基准上的性能见表7与图2。")
    table("表7 各检索方法在专家基准上的性能对比",
          ["方法", "NDCG@3", "Recall@3", "NDCG@5", "Recall@5", "MeanScore@3"],
          [["Random", "0.6214", "15.56%", "0.5825", "26.67%", "0.667"],
           ["未标准化统计", "0.5394", "17.78%", "0.5882", "22.22%", "0.578"],
           ["GNN", "0.5947", "11.11%", "0.6025", "20.00%", "0.778"],
           ["Hybrid", "0.7135", "15.56%", "0.7399", "31.11%", "0.800"],
           ["统计+欧氏距离", "0.7182", "28.89%", "0.6985", "37.78%", "1.000"],
           ["统计+余弦相似度", "0.7214", "26.67%", "0.6952", "37.78%", "0.889"],
           ["统计+曼哈顿距离", "0.7458", "35.56%", "0.7539", "37.78%", "1.111"]],
          bold_rows=(6,),
          note="注：粗体为最优值。")
    figure(os.path.join(FIG, "fig2_methods.png"), "图2 各检索方法性能对比", 14.5)
    normal("结果显示：（1）标准化统计描述子明显优于未标准化的原始统计（NDCG@3为0.721对0.539），说明"
           "标准化是平衡各维度贡献的必要步骤；（2）三种距离度量中，曼哈顿距离全面最优（NDCG@3为0.746，"
           "Recall@3为35.6%），余弦与欧氏接近，表明距离度量与结构描述子之间存在明显的耦合效应，需要通过"
           "实验确定；（3）本文采用的、以结构体系分类为训练目标的GNN表征路线在全部指标上最弱（Recall@3"
           "仅11.1%），Hybrid融合路线优于GNN但仍低于纯统计描述子，说明在当前任务设置下，该深度图表征"
           "未能为细粒度相似性排序带来增益，该现象的原因在第6.4节讨论；（4）Random的NDCG@3达0.621，"
           "验证了第4.3节的判断——NDCG分级增益使随机排序也能获得不低的数值，因此相对Random的提升应"
           "以Recall与MeanScore衡量：曼哈顿统计路线的Recall@3为Random的2.3倍，MeanScore@3由0.667提升至"
           "1.111，即前3个结果的平均参考价值得到实质性提高。")
    h2("5.3结构体系判别特征消融实验")
    normal("为回答“描述子的哪些信息层级最具判别力”，本文对描述子进行消融实验。需要明确该实验的口径："
           "消融实验在库内类型检索口径（Top-K类型检索准确率，留一式）下进行，度量的是描述子对结构体系"
           "的判别能力，而非专家结构相似性真值下的检索质量；后者由第5.2节的专家基准实验承担。单组实验"
           "仅使用某一组特征；累加实验按顺序逐组加入特征；留一组实验从完整描述子中去除某一组。结果见表8"
           "与图3。")
    table("表8 结构体系判别特征消融实验结果（Top-K准确率/%）",
          ["特征组合", "Top-1", "Top-3", "Top-5"],
          [["单用类型占比", "86.79", "93.40", "96.23"],
           ["单用面积占比", "80.19", "94.34", "95.28"],
           ["单用面积分布", "58.49", "88.68", "94.34"],
           ["单用空间统计", "37.74", "71.70", "84.91"],
           ["类型占比", "86.79", "93.40", "96.23"],
           ["类型占比+面积占比", "88.68", "96.23", "96.23"],
           ["类型占比+面积占比+面积分布", "87.74", "97.17", "98.11"],
           ["完整描述子", "87.74", "94.34", "98.11"],
           ["完整−类型占比", "84.91", "94.34", "96.23"],
           ["完整−面积占比", "87.74", "95.28", "97.17"],
           ["完整−面积分布", "85.85", "97.17", "98.11"],
           ["完整−空间统计", "87.74", "97.17", "98.11"]])
    figure(os.path.join(FIG, "fig3_ablation.png"), "图3 结构体系判别特征消融实验结果", 14.0)
    normal("消融结果表明：（1）在当前案例库与评价设置下，构件类型占比是最具判别力的特征组，单组Top-1"
           "即达86.8%，且去除该组时性能降幅最大（87.7%降至84.9%）；这与结构设计中“先看体系、再看细部”"
           "的工程认知一致，也与标准约束下同体系案例构件构成趋同的特点相符[4-5]；（2）面积信息（面积占比"
           "与面积分布）具有一定的增量贡献，与类型占比形成互补；（3）当前的空间统计量（最近邻距离均值与"
           "标准差）未表现出明显的独立增益，应当强调，这只能说明这两个统计量的表达粒度不足以有效区分"
           "同体系内的布置差异，并不能说明空间布置对结构相似性不重要，更精细的空间描述是后续改进方向。")
    h2("5.4按结构类型的专家检索性能与GNN增量价值")
    normal("表9给出两条主要路线按结构类型拆分的专家检索性能。")
    table("表9 按结构类型的专家检索性能（NDCG@3）",
          ["类型", "统计描述子", "GNN"],
          [["剪力墙", "0.861", "0.658"],
           ["框剪", "0.677", "0.426"],
           ["框架", "0.626", "0.700"]])
    figure(os.path.join(FIG, "fig5_tsne.png"), "图4 GNN嵌入的t-SNE分布（仅用于可视化）", 10.5)
    normal("统计描述子在剪力墙与框剪两类上均领先，而GNN在框架类反超。进一步分析候选覆盖发现：相比统计"
           "路线，GNN的Top-3新增35个候选，其中3个被专家评为≥3分、1个被评为4分，且这些高分增量候选全部"
           "出现在框架类查询。其原因在于：框架类图纸高度同质，GNN嵌入的类内两两余弦相似度几乎完全一致"
           "（均值0.999、标准差0.0013），表征已陷入类内坍缩，其排序接近同质样本间的任意扰动，仅在个别"
           "案例上恰好与专家偏好吻合；而统计描述子的类内相似度标准差保持在0.20～0.43，仍具有可区分度。"
           "这一观察说明，GNN可作为同质类型的候选补充，但不能作为主要排序依据；将两条路线的结果简单"
           "并集重排也未超过纯统计路线，表明融合的关键在于重排策略而非候选合并。图4的t-SNE可视化[19]"
           "直观显示了三类结构在GNN嵌入空间中的分离情况，仅作为辅助观察，不作为评价指标。")
    h2("5.5同项目与跨项目检索")
    normal("案例检索的真实价值在于从当前工程之外的历史项目中找到可参考案例。本文将每个查询的候选按"
           "项目归属拆分为同项目与跨项目子集分别评价（同项目子集仅包含存在同项目正样本的6个查询，"
           "跨项目子集包含8个查询），结果见表10与图5。")
    table("表10 同项目与跨项目检索性能对比",
          ["子集", "方法", "NDCG@3", "NDCG@5", "Recall@5"],
          [["同项目（n=6）", "统计描述子", "0.961", "0.961", "100%"],
           ["同项目", "GNN", "0.954", "0.954", "100%"],
           ["同项目", "Hybrid", "0.961", "0.961", "100%"],
           ["跨项目（n=8）", "统计描述子", "0.398", "0.501", "31.25%"],
           ["跨项目", "GNN", "0.187", "0.273", "12.50%"],
           ["跨项目", "Hybrid", "0.386", "0.456", "31.25%"]])
    figure(os.path.join(FIG, "fig4_cross_project.png"), "图5 同项目与跨项目检索性能对比", 12.5)
    normal("同项目子集任务容易，三种方法均接近饱和（NDCG@5≥0.95）；跨项目子集难度明显上升，各方法"
           "总体水平下降。在本研究专家样本的跨项目子集中（n=8），统计描述子（NDCG@5=0.501）明显高于"
           "GNN（0.273）与Hybrid（0.456），显示出更好的跨项目检索表现。由于跨项目子集样本量有限，"
           "本文不将其推广为普适性结论，但跨项目检索正是工程应用的核心场景，该结果仍具有明确的实践"
           "参考意义。")
    h2("5.6典型案例分析")
    normal("为直观展示检索行为，图6给出三组典型案例的结构平面构成对比。图中每张图纸按解析管线的构件"
           "分类着色：黑色为剪力墙、灰色为短肢剪力墙、白框为柱。")
    figure(os.path.join(FIG, "fig6_cases.png"), "图6 典型检索案例的结构平面构成对比", 16.0)
    normal("案例（a）为高一致案例：查询湘湖里_框架_H3的统计路线Top-1与Top-2候选（H8、H9）均获专家3分，"
           "GNN路线Top-1候选（H4）获4分，检索前列与专家判断高度一致，说明描述子能够捕捉同类标准方案"
           "之间的真实相似性。案例（b）同时体现了跨项目相似与弱标签失效：查询景芳三堡_框剪_B的跨项目"
           "候选勾庄好运来_框剪_20获专家3分，而同项目候选景芳三堡_框剪_C仅获1分——项目归属并不能保证"
           "结构相似；值得注意的是，统计路线Top-1候选西大门_框剪_2被专家评为0分。案例（c）进一步展示了"
           "这类误判：查询苏州华贸_剪力墙_10A的统计路线Top-1候选（勾庄好运来_剪力墙_8）仅获1分，而专家"
           "认为最有参考价值的苏州华贸_剪力墙_6B排在统计路线第5位。这些案例表明，当前11维全局统计描述子"
           "能够稳定识别“体系相同、构成接近”的案例，但对构件拓扑连接与局部布置差异尚不敏感，这正是"
           "后续引入更精细空间与拓扑描述的方向。")

    # ================= 6 讨论 =================
    h1("6 讨论")
    h2("6.1方法定位：结构相似性的量化探索")
    normal("本文工作的核心不是提出某个新算法，而是对一个此前缺少明确量化方法的问题——“建筑结构图纸"
           "之间如何定义和评价结构相似”——给出了一条可复现的技术路径：以工程先验（结构体系）约束检索"
           "空间，以工程语义（构件组成与尺度）描述结构，以专家评分定义相关性，并以项目身份置换检验保证"
           "检索过程不受项目信息影响。本文的全部结论均限定在当前数据集与实验设置之内，其价值在于方法"
           "框架与评价体系本身，后者在案例库与标注规模扩大时可直接沿用。")
    h2("6.2工程语义统计描述子为何有效、类型占比为何最具判别力")
    normal("统计描述子的有效性源于其与结构工程认知的对齐：设计人员判断结构相似性时，首先关注结构体系"
           "及其构件构成，而11维描述子正是按这一认知层级组织的。类型占比的最强判别力则有明确的工程依据："
           "我国建筑抗震设计相关标准对各结构体系的适用高度、构件构成与抗震措施有明确规定[4-5]，使同一"
           "体系内的案例在构件构成上天然趋同，类型占比因此成为天然的强特征。另一方面，类型占比与面积"
           "占比存在一定冗余——某类构件数量占比高时，其面积占比往往也高，这与累加消融中类型占比基础上"
           "加入面积占比提升有限的现象一致。")
    h2("6.3当前空间特征的表达局限")
    normal("消融实验中空间统计未表现出独立增益，本文对此持谨慎解释：当前仅使用最近邻距离的均值与标准差"
           "两个全局统计量，其表达粒度不足以有效区分同体系内的布置差异，而不能据此认为空间布置对结构"
           "相似性不重要。案例（c）中的误判也提示，全局统计量难以刻画构件拓扑连接与局部布置。后续将探索"
           "空间网格编码、构件邻接图匹配等更精细的空间描述。")
    h2("6.4GNN表征为何未表现出优势")
    normal("本文采用的、以结构体系分类为训练目标的GNN表征路线未表现出优势，其主要原因是训练目标与检索"
           "任务的错位：交叉熵分类监督优化的是类型间可分性，其结果是类内嵌入被压缩到极小区域（三类类内"
           "两两余弦相似度均值达0.80～0.999），形成类内表征坍缩，无法为第二阶段类内排序提供细粒度区分。"
           "这一观察并非对深度表征学习的否定：在框架类中GNN仍提供了个别高分增量候选，且106张图纸的样本"
           "规模也限制了高维嵌入的学习。以相似性为目标的度量学习或对比学习训练方式[15-17]，配合更大规模"
           "的专家标注，值得在后续工作中检验。")
    h2("6.5方法局限性")
    normal("本研究存在以下局限：（1）数据规模有限（106张图纸、20个项目），结论适用于当前案例库，尚不"
           "具备普适算法意义上的代表性；（2）专家标注由单一工程师完成，未进行多专家一致性检验（如加权κ"
           "系数），后续应增加至少一名结构工程师对部分样本复评；（3）空间统计特征表达粒度有限；（4）构件"
           "图以质心KNN建边，未利用墙体与柱之间实际的传力路径信息；（5）曼哈顿距离的优越性是当前数据集上"
           "的实验观察而非普适定理，距离度量与特征表示存在耦合，描述子变化时需要重新验证。")

    # ================= 7 结论 =================
    h1("7 结论")
    normal("本文针对建筑结构历史图纸的结构相似案例检索问题，提出了结构体系约束的两阶段结构相似性检索"
           "方法，构建了结构统计描述子与专家评价基准。在当前数据集与实验设置下，主要结论如下：")
    normal("1）按“结构体系—构件权重—尺寸规模—尺寸离散—空间布局”信息层级构建的11维结构统计描述子，"
           "能够定量刻画结构的全局特征；库内留一检索式结构体系识别Top-1准确率为87.7%；“先判型、后类内"
           "检索”的两阶段框架符合工程认知习惯，避免了跨结构体系的误匹配。")
    normal("2）特征标准化是描述子发挥作用的关键步骤；标准化描述子与曼哈顿距离的组合在专家基准上取得"
           "最佳检索性能（NDCG@3=0.746，Recall@3=35.6%，MeanScore@3=1.111），且距离度量的选择与描述子"
           "存在明显耦合。")
    normal("3）在当前案例库与评价设置下，构件类型占比是最具判别力的特征组；当前空间统计量的表达粒度"
           "不足以区分同体系内的布置差异，是后续改进方向。")
    normal("4）“同项目”只能作为弱标签：同项目案例对平均专家评分1.58，跨项目对仅0.55，且一半同项目对"
           "被判为不相似、11%的跨项目对被认为有参考价值；本文建立的专家基准与“检索项目无关、评估项目"
           "感知”原则（经项目名置换检验验证）为该方向研究提供了评价基础。在本研究专家样本的跨项目子集中，"
           "统计描述子（NDCG@5=0.501）明显高于GNN表征（0.273），显示出更好的跨项目检索表现。")
    normal("5）本文采用的、以分类为目标训练的GNN表征路线未表现出优势，类内坍缩使其无法支持细粒度类内"
           "排序，但在高度同质的框架类中仍能提供少量高分增量候选；面向相似性目标的深度表征学习方法值得"
           "进一步研究。")
    normal("需要指出，本文数据集规模（106张图纸、20个项目）与专家标注规模（171对、单一专家）仍然有限，"
           "上述结论适用于当前数据与实验设置；本文方法框架与评价体系在案例库与标注规模扩大时仍可直接"
           "沿用。")

    # ================= 参考文献 =================
    h1("参考文献")
    ref("[1] KOLODNER J L. Case-based reasoning [M]. San Francisco: Morgan Kaufmann, 1993.")
    ref("[2] AAMODT A, PLAZA E. Case-based reasoning: foundational issues, methodological variations, "
        "and system approaches [J]. AI Communications, 1994, 7(1): 39-59.")
    ref("[3] WATSON I. Applying case-based reasoning: techniques for enterprise systems [M]. San "
        "Francisco: Morgan Kaufmann, 1997.")
    ref("[4] GB 50011—2010 建筑抗震设计规范（2016年版） [S]. 北京: 中国建筑工业出版社, 2016.")
    ref("[5] JGJ 3—2010 高层建筑混凝土结构技术规程 [S]. 北京: 中国建筑工业出版社, 2011.")
    ref("[6] EASTMAN C, TEICHOLZ P, SACKS R, et al. BIM handbook: a guide to building information "
        "modeling for owners, designers, engineers, contractors, and facility managers [M]. 3rd ed. "
        "Hoboken: John Wiley & Sons, 2018.")
    ref("[7] DATTA R, JOSHI D, LI J, et al. Image retrieval: ideas, influences, and trends of the "
        "decade [J]. ACM Computing Surveys, 2008, 40(2): 5:1-5:60.")
    ref("[8] LATECKI L J, LAKÄMPER R, ECKHARDT T. Shape descriptors for non-rigid shapes with a "
        "single closed contour [C]//Proceedings of IEEE Conference on Computer Vision and Pattern "
        "Recognition (CVPR). Hilton Head: IEEE, 2000: 424-429.")
    ref("[9] FIX E, HODGES J L. Discriminatory analysis: nonparametric discrimination, consistency "
        "properties [R]. Randolph Field: USAF School of Aviation Medicine, 1951.")
    ref("[10] WU Z, PAN S, CHEN F, et al. A comprehensive survey on graph neural networks [J]. IEEE "
        "Transactions on Neural Networks and Learning Systems, 2021, 32(1): 4-24.")
    ref("[11] KIPF T N, WELLING M. Semi-supervised classification with graph convolutional networks "
        "[C]//Proceedings of the 5th International Conference on Learning Representations (ICLR). "
        "Toulon: OpenReview.net, 2017.")
    ref("[12] HAMILTON W L, YING R, LESKOVEC J. Inductive representation learning on large graphs "
        "[C]//Advances in Neural Information Processing Systems 30. Long Beach: Curran Associates, "
        "2017: 1024-1034.")
    ref("[13] VELIČKOVIĆ P, CUCURULL G, CASANOVA A, et al. Graph attention networks [C]//Proceedings "
        "of the 6th International Conference on Learning Representations (ICLR). Vancouver: "
        "OpenReview.net, 2018.")
    ref("[14] XU K, HU W, LESKOVEC J, et al. How powerful are graph neural networks? [C]//Proceedings "
        "of the 7th International Conference on Learning Representations (ICLR). New Orleans: "
        "OpenReview.net, 2019.")
    ref("[15] KULIS B. Metric learning: a survey [J]. Foundations and Trends in Machine Learning, "
        "2013, 5(4): 287-364.")
    ref("[16] SCHROFF F, KALENICHENKO D, PHILBIN J. FaceNet: a unified embedding for face recognition "
        "and clustering [C]//2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). "
        "Boston: IEEE, 2015: 815-823.")
    ref("[17] CHOPRA S, HADSELL R, LECUN Y. Learning a similarity metric discriminatively, with "
        "application to face verification [C]//2005 IEEE Computer Society Conference on Computer "
        "Vision and Pattern Recognition (CVPR). San Diego: IEEE, 2005: 353-360.")
    ref("[18] JÄRVELIN K, KEKÄLÄINEN J. Cumulated gain-based evaluation of IR techniques [J]. ACM "
        "Transactions on Information Systems, 2002, 20(4): 422-446.")
    ref("[19] VAN DER MAATEN L, HINTON G. Visualizing data using t-SNE [J]. Journal of Machine "
        "Learning Research, 2008, 9: 2579-2605.")

    # ================= 作者信息 =================
    ref("第一作者：钱路曦，男，硕士，工程师，浙江绿城建筑设计有限公司")
    ref("Email：qianluxi@gad.com.cn")
    ref("联系人：钱路曦")
    ref("基金资助项目名称及项目号：无")
    ref("文中图1～图6均为作者自绘，表1～表10均为实验结果统计")

    doc.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
