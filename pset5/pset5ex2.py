# curvefit .py Problem Sheet 5 example
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


def hello():
    def subfunc():
        pass
    self.x = subfunc


with open("pset5ex2.py") as fp:
    for i, line in enumerate(fp):
        if "\xe2" in line:
            print i, repr(line)
