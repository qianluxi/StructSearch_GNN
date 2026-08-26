from dataclasses import dataclass
from typing import Optional, Tuple, TypedDict

class NodeDict(TypedDict):
    centroid: Tuple[float, float]
    area: float
    Lx: float
    Ly: float
    long_leg_length: float
    long_leg_thickness: float
    r: float
    type: int


# node = {
#     "centroid": (x, y),
#     "area": float,
#     "Lx": float,
#     "Ly": float,
#     "long_leg_length": float,
#     "long_leg_thickness": float,
#     "r": float,
#     "type": int  # 0=Column,1=ShortWall,2=Wall
# }


@dataclass(frozen=True)
class DrawingMeta:
    """图纸元数据：项目与图纸标识（仅用于评估/展示，绝不参与检索）。"""

    project_id: str
    drawing_id: str
    structure_type: Optional[int] = None


def parse_drawing_meta(filename, structure_type=None):
    """从文件名解析项目/图纸标识。

    项目名 = 第一个下划线前的部分；该信息只作为评估（同项目/跨项目分析）
    与结果展示的元数据，检索阶段不得使用。

    Args:
        filename: 如 "勾庄好运来_框剪_17.dxf"
        structure_type: 可选，结构类型 id（0/1/2）

    Returns:
        DrawingMeta
    """
    project_id = filename.split("_", 1)[0]
    return DrawingMeta(
        project_id=project_id,
        drawing_id=filename,
        structure_type=structure_type,
    )
