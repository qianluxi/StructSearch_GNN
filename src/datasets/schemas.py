from typing import TypedDict, Tuple

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