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
    '''Initilise the animated line.
    '''
    print(0)
    line.set_data([], [])
    return line,

def animate(i):
    '''update the animated line with a square wave plot.
    n=i.
    '''
    print(i)
    line.set_data(x, square(x, i, L=np.pi))
    ax.set_title('n={}'.format(i))
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=50, interval=20, blit=False) #bilt turned off as it crashes on OSX
plt.show()


############## EX2

# curvefit.py Problem Sheet 5 example
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fitshow(x, y, popt, yn=None, yerr=None, func=None):
    '''Shows the plot with the best fit line.
    '''
    print 'Parameters : ', popt
    print 'Covariance : ', pcov
    # graphical output of results
    fig=plt.figure()
    plt.scatter(x,y, label='data')
    if yn is not None:
        if yerr is not None:
            plt.errorbar(x, yn, yerr=yerr, label='data + noise', fmt='o', color='r')
        else:
            plt.scatter(x,yn, color='r', label='data + noise')
    if func:
        plt.plot(x, func(x, *popt), color='green', label='best fit')
    plt.legend()
    plt.show()


def expfunc(x, a, b, c):
    return a*np.exp(-b*x) + c # function to generate data for curve fit
# The first part of this program generates a set of data that follows a particular functional form that allows us to test the curvefit routine

def sinfunc(x, a, b):
    '''Scipy helper func for fitting to trig functions.
    '''
    return a*np.sin(b*x)

# Now that we have our data we can attempt to fit to the curve
x = np.linspace(0, 4, 50)
y = expfunc(x, 3.0, 1.3, 5)
yn = y + 0.2*np.random.normal(size=len(x)) # adding some noise to the data points
popt, pcov = curve_fit(expfunc, x, yn) # performing curve fit, and returning parameters
fitshow(x, y, yn=yn, popt=popt, func=expfunc)

# add a std err approximation to bring the fit closer.
yerr = np.sqrt(np.abs(yn - y))
popt, pcov = curve_fit(expfunc, x, yn, sigma=yerr) # performing curve fit, and returning parameters
fitshow(x, y, yn=yn, popt=popt, yerr=yerr, func=expfunc)


pset5dat = np.loadtxt('fitting_ProblemSheet5data.dat').reshape((-1))
x5, y5 = pset5dat[::2], pset5dat[1::2] # take x, y axes from data
popt, pcov = curve_fit(sinfunc, x5, y5) # performing curve fit, and returning parameters
fitshow(x5, y5, popt=popt, func=sinfunc)


# strip PDF badness
with open("pset5ex2.py") as fp:
    for i, line in enumerate(fp):
        if "\xe2" in line:
            print i, repr(line)
