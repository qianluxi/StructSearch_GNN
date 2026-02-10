import ezdxf
import numpy as np
from shapely.geometry import Polygon
import math

def get_aspect_ratio_and_area(polygon):
    """
    计算多边形的面积和长宽比。
    长宽比定义为：最小外接矩形的长边 / 短边。
    """
    if not polygon.is_valid or polygon.is_empty:
        return 0.0, 1.0

    area = polygon.area
    
    # 获取最小外接矩形 (Minimum Rotated Rectangle)
    rect = polygon.minimum_rotated_rectangle
    
    # 获取矩形的四个顶点
    # shapely 的 bounds 只是轴对齐矩形，我们要用 exterior.coords 获取旋转后的
    try:
        x, y = rect.exterior.coords.xy
        # 计算边长：由于是矩形，计算相邻两点的距离即可
        # 取前3个点算出2条边 (第0点和第1点，第1点和第2点)
        p0 = np.array([x[0], y[0]])
        p1 = np.array([x[1], y[1]])
        p2 = np.array([x[2], y[2]])
        
        d1 = np.linalg.norm(p0 - p1)
        d2 = np.linalg.norm(p1 - p2)
        
        edge_lengths = [d1, d2]
        
        l = max(edge_lengths)
        w = min(edge_lengths)
        
        if w < 1e-6: # 防止除以0
            ratio = 1.0
        else:
            ratio = l / w
            
    except Exception as e:
        # 如果几何计算出错（比如退化成一条线），返回默认值
        ratio = 1.0
        
    return area, ratio

def parse_dxf_file(filepath):
    """
    读取DXF文件，提取所有闭合多段线（代表柱、墙等构件）。
    返回: 
        polygons: shapely Polygon 对象列表
        centroids: 质心坐标列表 [(x, y), ...]
        features: 原始特征列表 [[area, ratio], ...]
    """
    try:
        doc = ezdxf.readfile(filepath)
    except IOError:
        print(f"Error: Cannot read file {filepath}")
        return [], [], []
    except ezdxf.DXFStructureError:
        print(f"Error: Invalid DXF file {filepath}")
        return [], [], []

    msp = doc.modelspace()
    polygons = []
    centroids = []
    raw_features = [] # [area, aspect_ratio]

    # 遍历所有 LWPOLYLINE (轻量多段线)
    for entity in msp.query('LWPOLYLINE'):
        if entity.is_closed:
            # 获取点坐标 (只要x, y，不要z)
            points = list(entity.get_points(format='xy'))
            
            # 过滤掉点数过少或面积过小的噪点
            if len(points) < 3:
                continue
            
            poly = Polygon(points)
            
            # 简单的去噪：面积太小的可能是文字框或杂线
            if poly.area < 1000: # 这里的阈值取决于你的DXF单位（毫米或米），假设是毫米，1000mm²很小
                continue

            area, ratio = get_aspect_ratio_and_area(poly)
            
            polygons.append(poly)
            centroids.append([poly.centroid.x, poly.centroid.y])
            raw_features.append([area, ratio])

    return polygons, np.array(centroids), np.array(raw_features)