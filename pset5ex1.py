import numpy as np
import matplotlib.pyplot as plt

def f_series(n, a_func, b_func, l=1000):
    arr = np.zeros(l)
    for i in range(n):
        arr += a_func(i) + b_func(i)
    return arr

def squ_wave(n, l=1000):
    b_func = lambda i: 2 / (i * np.pi) if i%2 else 0
    a_func = lambda i: 0 # odd func, no even coeffs

    return f_series(n, a_func, b_func, l)

plt.plot(squ_wave(10))
plt.show()
