"""
problem set 6 exersise 3

Thomas Parks
"""
from scipy.integrate import quad
from scipy.special import hermite
from scipy.optimize import curve_fit
from scipy.misc import factorial
from functools import partial
import numpy as np
from numpy import power, sqrt, exp, pi
from sys import float_info

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import multiprocessing as mp
pool = mp.Pool(processes=8)

a = 1

def parabolic_phi(x, n, a=a):
    return hermite(n)(x/a) * exp(-x**2/(2*a**2)) / sqrt(2**n * factorial(n) * sqrt(pi*a**2))

# useful funcs to have around for the multiprocessing later
phin = [partial(parabolic_phi, n=n) for n in range(100)]

def Pcl(x, x0=6):
    """classical probability for a oscillator.
    """
    return 1/(pi*sqrt(x0**2 - x**2))

def integralphi(maxx, n):
    """returns the integral of phi^2 for the range -maxx, maxx.
    """
    return quad(lambda x: parabolic_phi(x, n)**2, -maxx, maxx)[0]

def expctationvalue(wavef=parabolic_phi,
                    wavefargs={},
                    operator=lambda x: x,
                    maxx=10*a,
                    quadargs={}):
    """find the expectation value of an operator function on a
    QM wavefnction.

    wavefargs: argument list for the wavefunc
    quadargs: argurements for the integration
    returns: real number
    """
    ev, err = quad(lambda x: operator(x)*wavef(x, **wavefargs)**2,
                   -maxx, maxx, **quadargs)
    return ev

def main():
    xaxis = range(50)
    for n in range(5):
        print 'hermite', n, '=' # leave a line
        print hermite(n) # disply the hermetian polynomials
        print " this is the same as the tabulated coeff."
        print "For phi{}, over -5a, 5a equals {}".format(n,
                                                         quad(lambda x: parabolic_phi(x, n)**2,
                                                              -a*5, a*5)[0]
                                                        )

    # get the axes for the plots
    fig, (phiax, probax) = plt.subplots(2, 1)
    figx = np.linspace(-6, 6, 1000)
    for n in range(10+1):
        phiax.plot(figx, [phin[n](x)+n for x in figx], label="n={}".format(n))
        probax.plot(figx, [phin[n](x)**2+n for x in figx], label="n={}".format(n))

    for ax in [phiax, probax]:
        ax.set_xlim(-6, 8)
        ax.legend()
    phiax.set_ylabel("$\Psi _n(x)$")
    probax.set_ylabel("$\left| \Psi _n(x)\\right|^2 $")
    probax.set_xlabel("$x$")
    fig.suptitle("Wavefuncs for a potential well")
    plt.show()

    # check orthoganality
    Qnumbers = np.linspace(0, 50, 20, dtype=np.int16)
    for QN in Qnumbers:
        ev = expctationvalue(wavefargs={'n':QN}, quadargs={'limit':n*30})
        if abs(ev) > float_info.epsilon:
            print "orthogonal FAILURE for QN={}".format(QN)

    # find the uncirtancty in the phi's
    for QN in [4, 7, 10]:
        obsx = expctationvalue(wavefargs={'n':QN}, quadargs={'limit':n*30})
        obsx2 = expctationvalue(wavefargs={'n':QN}, quadargs={'limit':n*30},
                                operator= lambda x: x**2)


        print "{}: obs X {} obs X^2 {} s.d. {}".format(
                                                   QN,
                                                   obsx,
                                                   obsx2,
                                                   sqrt(obsx2 - obsx**2)
                                                )
        print "dx formula gives {}".format(sqrt( (2*QN+1)/2 ))
        print "x0 = sqrt(2)*dx = {}, test {}".format(sqrt(2*(obsx2 - obsx**2)),
                                                     sqrt(2*QN+1))




    fig = plt.figure()
    ax = plt.axes(xlim=(-8, 8), ylim=(0, 1))

    line, = ax.plot([], [], 'r-')

    def animate(i):
        print(i)
        line.set_data(figx, [phin[i](x)**2 for x in figx])
        ax.set_title('n={}'.format(i))
        return line,

    anim = animation.FuncAnimation(fig, animate,
                                   frames=20, interval=20, blit=False) #bilt turned off as it crashes on OSX

    # PLOT the comp from QM to classical probability
    ax.plot(figx, [Pcl(x) for x in figx], label="classical probability",
             linewidth=2.0)
    ax.set_ylabel("$\left| \Psi _n(x)\\right|^2 $")
    ax.set_xlabel("$x$")
    ax.set_title("QM and classical comp.")
    plt.figtext(2., .8, "as n goes to high, the QM version goes to the classical.")

    plt.show()

if __name__ == "__main__":
    main()
