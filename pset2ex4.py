from numpy.linalg import norm
import numpy as np
import timeit
import math
import sys
import cProfile

sys.setrecursionlimit(sys.getrecursionlimit() * 10)


def pathlen(points):
    length = 0
    for i, point in enumerate(points[1:]):
        length += norm(points[i] - point)
    return length


def pathlen_xys(xs, ys):
    return pathlen([np.array(v) for v in zip(xs, ys)])


# recursive version of the function to calculate a path length
def functional_path_length(xs, ys, x2=False, y2=False):
    x2 = x2 if x2 else xs[0]
    y2 = y2 if y2 else ys[0]
    if len(xs) > 0:
        return math.sqrt((xs[0] - x2)**2 + (ys[0] - y2)**2) + functional_path_length(xs[1:], ys[1:], xs[0], ys[0])
    else:
        return 0


xsn = np.random.normal(size=5000)
ysn = np.random.normal(size=5000)
points = [np.array(v) for v in zip(xsn, ysn)]
xs = list(xsn)
ys = list(ysn)

cProfile.run('pathlen(points)')
cProfile.run('functional_path_length(xs, ys)')

'''
n1 = timeit.timeit('pathlen(points)', setup='from __main__ import pathlen, points', number=1000)
n2 = timeit.timeit('functional_path_length(xs, ys)', setup='from __main__ import functional_path_length, xs, ys', number=1000)

print('tom', n1, 'will', n2)
'''


def pi(n):
    xpoints = np.linspace(-1, 1, n)
    ypoints = np.sqrt(1 - xpoints**2)
    points = [np.array(v) for v in zip(xpoints, ypoints)]
    return pathlen(points)
