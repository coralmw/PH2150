import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# we use the f_adder, but partially evaluate it

# omg this would be really easy in a languade with partial evaluation

# factory function
class f_adder(object):

    # index needs to go first so that it can be positionally indexed
    # when partially evaulated
    def __init__(self, vector, multiplier, x_axis, index):
        self.m = multiplier
        self.v = vector
        self.i = index

    def __call__(self, array):
        print(type(array))
        array += self.v(array)*self.m(self.i)


b_func = partial(f_adder, np.sin)
a_func = partial(f_adder, np.cos)

def f_series(n, a_func, b_func, l=1000):
    x = np.linspace(-np.pi, np.pi, l)
    L = 2*np.pi

    arr = a_func(0) / 2
    for i in range(1, n+1):
        trigmult = i*np.pi/L
        arr += a_func(i) * np.cos(trigmult*x)
        arr += b_func(i) * np.sin(trigmult*x)
    return x, arr

def squ_wave(n, l=1000):
    x = np.linspace(-np.pi, np.pi, l)
    b_func = lambda i: 4/np.pi * (1/i if i%2==1 else 0)
    a_func = lambda i: 0
    return f_series(n, a_func, b_func, l)

def squ_w(x, nmax):
    L = 2*np.pi
    v = 0
    for n in range(1, nmax):
        if n%2 == 1:
            v += np.sin(n*np.pi*x/L)/n
    return v*4/np.pi



x = np.linspace(-np.pi, 3*np.pi, 1000)
v = [squ_w(xp, 1000) for xp in x]

plt.plot(x, v)
plt.show()
