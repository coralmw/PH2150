P10 = ["hydrogen", "helium", "lithium", "beryllium", "boron", "carbon",
       "nitrogen", "oxygen", "fluorine", "neon"]
P20 = ["sodium", "magnesium", "aluminum", "silicon", "phosphorus", "sulfur",
       "chlorine", "argon", "potassium", "calcium"]
P30 = ["scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
       "cobalt", "nickel", "copper", "zinc"]

Elems = P10 + P20
for e in P30:  # append eac h elem to the end of the list
    Elems.append(e)

if __name__ == '__main__':
    print(len(Elems))
    print(Elems[22])



def Elems_eq4():
    '''
    Gets the 4 letter elements from the global elems.
    '''
    for e in Elems:
        if len(e) == 4:
            yield e # returns each 4 letter elem
    raise StopIteration

if __name__ == '__main__':
    print([e for e in Elems if e[0] == 's']) # list comprehension: checks the first letter is s
    print(list(Elems_eq4())) # a genrator of 4-letter elems


import numpy as np
from numpy import dot, cross
from numpy.linalg import norm

def area(x1, x2, y1, y2): # 2d triangle
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    return 0.5 * dx * dy

def area_triangle(v1, v2, v3): # rearranges the vertacies into a sensible form for linalg
    return area_verts(np.array(v1+[0]),
                      np.array(v2+[0]),
                      np.array(v3+[0]))

def area_verts(p1, p2, p3):
    '''
    returns the area of a triangle.

    p(1-3): 3-tuples or 3-elem numpy arrays, the points of the triangle
    returns: the supreme type of type(p1-3), representing the size of triangle
    '''
    area = 0.5 * dot(p1, cross(p2, p3))
    return area


from numpy.linalg import norm
import numpy as np
import math
import sys

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

def pi(n):
    i = np.linspace(0, n, n+1)
    xpoints = 0.5*np.cos( 2*np.pi*i/n )
    ypoints = 0.5*np.sin( 2*np.pi*i/n )
    points = [np.array(v) for v in zip(xpoints, ypoints)] # Got to be a better way.
    return pathlen(points)

if __name__ == '__main__':
    for power in range(11):
        iterations = 2**power
        approxpi = pi(iterations)
        print('error with {} iterations is {}'.format(iterations, abs(np.pi - approxpi)))
