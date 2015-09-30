import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# we use the f_adder, but partially evaluate it

# omg this would be really easy in a languade with partial evaluation

# factory function
class f_adder(np.ndarray):

    # index needs to go first so that it can be positionally indexed
    # when partially evaulated
    def __init__(self, index, multiplier, vector):
        self.m = multiplier
        self.v = vector
        self.i = index

    def __radd__(self, array):
        print(type(array))
        return array + self.v(array)*self.m(self.i)


def f_series(n, a_func, b_func, l=1000):
    arr = np.zeros(l)
    print(type(arr))
    for i in range(n):
        arr += a_func(i)
        arr += b_func(i)
    return arr


def squ_wave(n, l=1000):
    b_func = partial(f_adder,
                     multiplier=lambda i: 2 / (i * np.pi) if i%2 else 0,
                     vector=np.sin)

    a_func = partial(f_adder,
                     multiplier=lambda i: 0,
                     vector=np.cos)

    return f_series(n, a_func, b_func, l)

plt.plot(squ_wave(1))
plt.show()
