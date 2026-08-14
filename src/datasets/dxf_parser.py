import os
import math
import logging
import ezdxf
from shapely.geometry import Polygon
from shapely.affinity import rotate


logger = logging.getLogger(__name__)


def _expand_insert(insert, depth=0):
    """展开 INSERT 块引用，返回其中的 LWPOLYLINE 实体（含嵌套块）。

    ezdxf 的 virtual_entities() 会把块内实体按插入变换到模型空间坐标；
    嵌套 INSERT 最多递归 4 层，防止异常文件造成死循环。
    """
    if depth >= 4:
        return []

    result = []
    block_name = getattr(insert.dxf, "name", "?")
    try:
        virtual = insert.virtual_entities()
    except Exception:  # noqa: BLE001 - 单个块展开失败不应中断解析
        logger.warning("展开 INSERT 失败（块 %s），跳过", block_name)
        return result

    for entity in virtual:
        if entity.dxftype() == "LWPOLYLINE":
            result.append(entity)
        elif entity.dxftype() == "INSERT":
            result.extend(_expand_insert(entity, depth + 1))
    return result


# ==========================================
# 读取 DXF
# ==========================================

from shapely.geometry import Polygon
import numpy as np

def load_polygons_from_dxf(path, close_threshold=1.0):  # 1mm 以内就认为闭合
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    polygons = []

    total_polyline = 0
    closed_count = 0
    skipped_count = 0
    force_closed_count = 0  # 强制闭合计数

    logger.info("开始解析 DXF: %s", path)

    polyline_entities = []
    for entity in msp:
        if entity.dxftype() == "LWPOLYLINE":
            polyline_entities.append(entity)
        elif entity.dxftype() == "INSERT":
            expanded = _expand_insert(entity)
            logger.debug(
                "展开 INSERT 块 %s，新增 %s 个 LWPOLYLINE",
                getattr(entity.dxf, "name", "?"),
                len(expanded),
            )
            polyline_entities.extend(expanded)

    for entity in polyline_entities:
        total_polyline += 1
        pts = [(p[0], p[1]) for p in entity]
        num_pts = len(pts)

        # 点数太少不能构成多边形
        if num_pts < 3:
            skipped_count += 1
            logger.debug("跳过：点数不足 %s", num_pts)
            continue

        # --------------------------
        # 原来的判断
        # --------------------------
        if entity.closed:
            closed_count += 1
            polygons.append(Polygon(pts))
            logger.debug("正常闭合多边形，点数: %s", num_pts)
        else:
            # --------------------------
            # 新增：强制判断近似闭合
            # --------------------------
            p_start = np.array(pts[0])
            p_end = np.array(pts[-1])
            dist = np.linalg.norm(p_start - p_end)  # 起点终点距离

            if dist < close_threshold:
                force_closed_count += 1
                polygons.append(Polygon(pts))
                logger.debug("强制闭合（距离差=%.2f），点数: %s", dist, num_pts)
            else:
                skipped_count += 1
                logger.debug("跳过不闭合（距离差=%.2f），点数: %s", dist, num_pts)

    logger.info(
        "解析结果汇总: 总 LWPOLYLINE=%s, 正常闭合=%s, 强制闭合=%s, "
        "跳过=%s, 最终 polygons=%s",
        total_polyline, closed_count, force_closed_count,
        skipped_count, len(polygons),
    )

    return polygons

# ==========================================
# 对齐 polygon
# ==========================================

def align_polygon(polygon):

    rect = polygon.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)

    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]

    angle = math.degrees(math.atan2(dy, dx))

    aligned = rotate(polygon, -angle, origin="centroid", use_radians=False)

    return aligned


# ==========================================
# 计算跨度
# ==========================================

def compute_spans(polygon):

    minx, miny, maxx, maxy = polygon.bounds

    Lx = maxx - minx
    Ly = maxy - miny

    return Lx, Ly


# ==========================================
# 提取边
# ==========================================

def extract_edges(polygon):
    """
    polygon: shapely Polygon
    return: list of edge dict
    """
    coords = list(polygon.exterior.coords)
    edges = []

    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        if length <= 1e-9:
            continue

        edges.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "dx": dx,
            "dy": dy,
            "length": length
        })

    return edges


# ==========================================
# 核心：长肢厚度计算（稳定工程版）
# ==========================================

def merge_parallel_edges(edges, direction, tol=1):
    """
    修复点：
    1. 使用量化 key 进行稳定分组
    2. 避免浮点误差导致分组随机波动
    3. 保持可重复性
    """

    groups = {}

    for e in edges:
        dx = e["dx"]
        dy = e["dy"]

        # 判断方向
        if direction == "horizontal":
            if abs(dy) > abs(dx):
                continue
            pos = e["y1"]
            minv = min(e["x1"], e["x2"])
            maxv = max(e["x1"], e["x2"])
        else:
            if abs(dx) > abs(dy):
                continue
            pos = e["x1"]
            minv = min(e["y1"], e["y2"])
            maxv = max(e["y1"], e["y2"])

        # ========= 核心修复 =========
        # 使用 round 量化，确保稳定
        key = round(pos / tol) * tol

        if key not in groups:
            groups[key] = {
                "pos": key,
                "segments": []
            }

        groups[key]["segments"].append((minv, maxv, e["length"]))

    merged = []

    for key in sorted(groups.keys()):
        group = groups[key]
        segments = group["segments"]

        segments.sort(key=lambda x: x[0])

        total_length = 0.0
        cur_start, cur_end, _ = segments[0]

        for s in segments[1:]:
            if s[0] <= cur_end + tol:
                cur_end = max(cur_end, s[1])
            else:
                total_length += cur_end - cur_start
                cur_start, cur_end = s[0], s[1]

        total_length += cur_end - cur_start

        merged.append({
            "pos": group["pos"],
            "min": min(s[0] for s in segments),
            "max": max(s[1] for s in segments),
            "length": total_length
        })

    return merged

def compute_r_from_long_leg(polygon, Lx, Ly, tol=1e-6, debug=True):

    if debug:
        logger.debug("开始计算 r: Lx=%s, Ly=%s", Lx, Ly)

    edges = extract_edges(polygon)

    if debug:
        logger.debug("总边数: %s", len(edges))

    if Lx >= Ly:
        span = Lx
        direction = "horizontal"
    else:
        span = Ly
        direction = "vertical"

    if debug:
        logger.debug("判定主方向: %s, span = %s", direction, span)

    merged_lines = merge_parallel_edges(edges, direction, tol)

    if debug:
        logger.debug("合并后平行线数量: %s", len(merged_lines))

    if len(merged_lines) < 2:
        if debug:
            logger.debug("平行线不足2条，返回0")
        return 0.0, 0.0

    merged_lines.sort(key=lambda x: x["length"], reverse=True)

    if debug:
        for i, line in enumerate(merged_lines):
            logger.debug("合并线 %s: pos=%s, len=%s", i, line["pos"], line["length"])

    best_thickness = None
    best_r = 0.0

    for i in range(len(merged_lines)):
        for j in range(i + 1, len(merged_lines)):

            line1 = merged_lines[i]
            line2 = merged_lines[j]

            if debug:
                logger.debug("检查边对 (%s,%s)", i, j)

            if abs(line1["pos"] - line2["pos"]) < tol:
                if debug:
                    logger.debug("  跳过：共线")
                continue

            overlap_min = max(line1["min"], line2["min"])
            overlap_max = min(line1["max"], line2["max"])
            overlap = max(0.0, overlap_max - overlap_min)

            min_len = min(line1["length"], line2["length"])
            overlap_ratio = overlap / min_len if min_len > 0 else 0

            if debug:
                logger.debug("  overlap = %s", overlap)
                logger.debug("  overlap_ratio = %s", overlap_ratio)

            if overlap < 0.3 * min_len:
                if debug:
                    logger.debug("  跳过：重叠不足30%")
                continue

            thickness = abs(line1["pos"] - line2["pos"])

            if debug:
                logger.debug("  thickness = %s", thickness)

            if thickness <= tol:
                if debug:
                    logger.debug("  跳过：厚度小于tol")
                continue

            if thickness >= span:
                if debug:
                    logger.debug("  跳过：厚度>=span")
                continue

            if debug:
                logger.debug("  有效厚度候选")

            if best_thickness is None or thickness < best_thickness:
                best_thickness = thickness
                best_r = span / thickness

                if debug:
                    logger.debug("  更新最佳厚度: %s, r=%s", best_thickness, best_r)

    if best_thickness is not None:
        if debug:
            logger.debug("最终选择厚度: %s, 最终 r: %s", best_thickness, best_r)
        return best_r, best_thickness
    else:
        if debug:
            logger.debug("没有找到有效厚度")
        return 0.0, 0.0
# ==========================================
# 分类（修改后）
# ==========================================
def classify_node(r, thickness):
    # 新规则：若长肢厚度 ≥ 310 且 长宽比 > 4，直接归为剪力墙
    if thickness >= 310 and r > 4:
        return "Wall"
    # 原规则保持不变
    elif r >= 8:
        return "Wall"
    elif r >= 4:
        return "ShortWall"
    else:
        return "Column"

# 找长肢的角度
def get_longest_edge_direction(polygon):
    """返回多边形中最长边的方向角度（度）"""
    edges = extract_edges(polygon)
    if not edges:
        return 0.0
    longest = max(edges, key=lambda e: e['length'])
    angle = math.degrees(math.atan2(longest['dy'], longest['dx']))
    return angle

# ==========================================
# 主函数
# ==========================================

def process_dxf_file(path):
    polygons = load_polygons_from_dxf(path)
    nodes = []

    for polygon in polygons:
        if polygon.area < 1e-4:
            continue
        aligned = align_polygon(polygon)
        Lx, Ly = compute_spans(aligned)
        r, thickness = compute_r_from_long_leg(aligned, Lx, Ly)
        if r is None:
            continue

        if thickness <= 1e-6:
            angle = get_longest_edge_direction(polygon)
            aligned2 = rotate(polygon, -angle, origin='centroid', use_radians=False)
            Lx2, Ly2 = compute_spans(aligned2)
            r2, thickness2 = compute_r_from_long_leg(aligned2, Lx2, Ly2)
            if thickness2 > 1e-6:
                aligned, Lx, Ly, r, thickness = aligned2, Lx2, Ly2, r2, thickness2

        node_type_str = classify_node(r, thickness)

        type_map = {
            "Column": 0,
            "ShortWall": 1,
            "Wall": 2
        }

        nodes.append({
            "centroid": aligned.centroid.coords[0],
            "area": aligned.area,
            "Lx": Lx,
            "Ly": Ly,
            "long_leg_length": max(Lx, Ly),
            "long_leg_thickness": thickness,
            "r": r,
            "type": type_map[node_type_str]
        })

    return nodes

