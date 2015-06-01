import numpy as np
from numpy.linalg import dot, cross, norm


def area(x1, x2, y1, y2):
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    return 0.5 * dx * dy


def area_verts(p1, p2, p3):
    area = 0.5 * dot(p1, cross(p2, p3))
    return area
