from numpy.linalg import norm
import numpy as np
import math
import sys
import cProfile

TIMING = False

# Thomas Parks version
def pathlen(points):
    """Finds the length of a line connecting the given points.

    points: [numpy vector, numpy vector, ...]
    returns: float of the length of the connecting path.
    """
    length = 0.
    for i, point in enumerate(points[1:]):
        length += norm(points[i] - point)
    return length

# This has the function signature you expect
def pathlen_xys(xs, ys):
    return pathlen([np.array(v) for v in zip(xs, ys)])


# recursive version of the function to calculate a path length
# Will Luckin's rubbish, stack buster version
# def functional_path_length(xs, ys, x2=False, y2=False):
#     x2 = x2 if x2 else xs[0]
#     y2 = y2 if y2 else ys[0]
#     if len(xs) > 0:
#         return math.sqrt((xs[0] - x2)**2 + (ys[0] - y2)**2) + functional_path_length(xs[1:], ys[1:], xs[0], ys[0])
#     else:
#         return 0


if TIMING:
    xsn = np.random.normal(size=5000)
    ysn = np.random.normal(size=5000)
    points = [np.array(v) for v in zip(xsn, ysn)]
    xs = list(xsn)
    ys = list(ysn)

    cProfile.run('pathlen(points)') # Mine is faster.
    cProfile.run('functional_path_length(xs, ys)')


def pi(n):
    i = np.linspace(0, n, n+1)
    xpoints = 0.5*np.cos( 2*np.pi*i/n )
    ypoints = 0.5*np.sin( 2*np.pi*i/n )
    points = [np.array(v) for v in zip(xpoints, ypoints)] # Got to be a better way.
    return pathlen(points)

for power in range(11):
    iterations = 2**power
    approxpi = pi(iterations)
    print('error with {} iterations is {}'.format(iterations, abs(np.pi - approxpi)))
