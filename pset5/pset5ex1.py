import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from functools import partial

def f_series(x, nmax, L=2*np.pi, a_func=lambda n: 0, b_func=lambda n: 0):
    '''
    returns a numpy array of the f summation over the x-array.

    x: numpy array, x-points
    L: x-range
    a,b_func: functions that evaluate to the coeff
    '''
    v = np.zeros_like(x) + a_func(0)/2
    for n in range(1, nmax+1):
        v += np.cos(n*np.pi*x/L)*a_func(n)
        v += np.sin(n*np.pi*x/L)*b_func(n)
    return v

# partailly evaluated fuctions for each kind of wave
square = partial(f_series,
                 a_func=lambda n: 0, # lambda funcs for the an bn
                 b_func=lambda n: 4/(np.pi*n) if n%2==1 else 0
                )

saw = partial(f_series,
              a_func=lambda n: 0 if n>0 else 1,
              b_func=lambda n: np.pi * (-2/(np.pi*n))
             )

# 1000 points over the range
x = np.linspace(0, 2*np.pi, 1000)

fig, (squplt, triplot) = plt.subplots(2, 1)

# plot the levels of iteration for each f(x)
labelstring = "n={}"
for n in [9, 99, 999]:
    squplt.plot(x, square(x, n, L=np.pi), label=labelstring.format(n))
    triplot.plot(x, saw(x, n, L=np.pi/2), label=labelstring.format(n))

# set the xlims slightly larger to make room for the label
for ax in [squplt, triplot]:
    ax.set_xlim(0, 2*np.pi+2)

squplt.legend()
triplot.legend()

plt.show()

# make the animation figure
fig = plt.figure()
ax = plt.axes(xlim=(0, 2*np.pi), ylim=(-2, 2))
line, = ax.plot([], [], 'r-')

def init():
    print(0)
    line.set_data([], [])
    return line,

def animate(i):
    print(i)
    line.set_data(x, square(x, i, L=np.pi))
    ax.set_title('n={}'.format(i))
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=50, interval=20, blit=False) #bilt turned off as it crashes on OSX
plt.show()
