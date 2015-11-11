from scipy.integrate import quad
from scipy.special import hermite
from functools import partial
import numpy as np

import matplotlib.pyplot as plt
from sys import float_info
import multiprocessing as mp

LengthScale=1.

# define the wavefunction
def psi_xnL(L, n, x):
    return np.sqrt(2./L) * np.sin(n*np.pi*x/L)

# set L=1
psi_xn = partial(psi_xnL, LengthScale)

def dx(n):
    expx, expxerr = quad(lambda x: psi_xn(n, x)*x*psi_xn(n, x), 0, LengthScale, limit=200)
    expx2, expx2err = quad(lambda x: psi_xn(n, x)*x*x*psi_xn(n, x), 0, LengthScale, limit=200)
    return np.sqrt(expx2 - expx**2)


def main1():
    xaxis = np.linspace(-LengthScale, LengthScale, 1000)

    fig, (sep, together) = plt.subplots(nrows=2, ncols=1)

    sep.plot(xaxis, psi_xn(3, xaxis), 'r', label="n=3")
    sep.plot(xaxis, psi_xn(4, xaxis), 'g', label="n=4")
    sep.legend()

    product34 = psi_xn(3, xaxis)*psi_xn(4, xaxis)
    together.plot(xaxis, product34)
    together.axhline(product34.mean(), color='r')

    plt.show()

    orth, ortherr = quad(lambda x: psi_xn(3, x)*psi_xn(4, x), 0, LengthScale)

    if abs(orth) < (float_info.epsilon + ortherr):
        print("the phi's are orthogonal")
    else:
        print("not orthogonal")

    nms = [(1, 2), (2, 3), (2, 2), (3, 3)]
    for n, m in nms:
        orth, ortherr = quad(lambda x: psi_xn(n, x)*psi_xn(m, x), 0, LengthScale)

        if abs(orth) < (float_info.epsilon + ortherr):
            print("the phi's {}, {} are orthogonal".format(n,m))
        else:
            print("the phi's {}, {} are NOT orthogonal, have value {}".format(n,m,orth))

    print("expectation value")
    for n in [5, 7, 9]:
        expx, expxerr = quad(lambda x: psi_xn(n, x)*x*psi_xn(n, x), 0, LengthScale)
        print("for psi{}, expx={}, L={}".format(n,expx,LengthScale))

    for n in [4, 7, 100]:
        expx, expxerr = quad(lambda x: psi_xn(n, x)*x*psi_xn(n, x), 0, LengthScale)
        expx2, expx2err = quad(lambda x: psi_xn(n, x)*x*x*psi_xn(n, x), 0, LengthScale)
        print("for psi{}, expx={}, x^2={}, L={}".format(n,expx,expx2,LengthScale))
        print("uncirt x={}".format(np.sqrt(expx2 - expx**2)))

    n = np.linspace(1, 30, 1000)

    pool = mp.Pool(processes=8)
    vals = pool.map(dx, n)
    plt.plot(n, vals)
    plt.axhline(LengthScale/np.sqrt(12))

    plt.plot(n, abs(LengthScale/np.sqrt(12) - vals))

    plt.show()



####### EX2


from scipy.integrate import quad
from scipy.special import hermite
from scipy.optimize import curve_fit
from functools import partial
import numpy as np
from numpy import power, sqrt
import multiprocessing as mp
pool = mp.Pool(processes=8)


import matplotlib.pyplot as plt

import unittest

from pset6ex1 import LengthScale, psi_xnL

# define the wavefunction
def A(x, L=1):
    '''wavefunc to model.
    x is the xpoint to calculate the value at.
    '''
    return power(x/L, 3) - 11.*np.power(x/L, 2)/7. + 4.*np.power(x/L, 1)/7.

def Anorm(x, L=1):
    return A(x, L)

def norm(wavefunc=A, L=1):
    integral, error = quad(lambda x: A(x)**2, 0, L)
    norm = 1/sqrt(integral)
    return norm

def cn(n, wavefunc=Anorm, sumfunc=psi_xnL, L=1, intlim=50):
    return quad(lambda x: psi_xnL(L,n,x)*norm(A)*A(x), 0, 1, limit=intlim)[0]

def cnsqu(n):
    return cn(n, intlim=5000)**2

def cnsqunsqu(n):
    return abs(cn(n))**2*n**2

def expfunc(x, a, b, c):
    """Function for curve fitting."""
    return a*np.exp(-b*x) + c # function to generate data for curve fit

# create the cn array, as it's a large calculation
nn = np.linspace(1, 50, 50)
cn_array = [cn(n) for n in nn]
popt, pcov = curve_fit(expfunc, nn, cn_array) # performing curve fit, and returning parameters

nnsqu = np.linspace(1, 500, 500, dtype=np.int32) # use a much larger array b/c convergance is slower
cnsqu_array = np.cumsum([cnsqu(n) for n in nnsqu])
poptcnsqu, pcovcnsqu = curve_fit(expfunc, nnsqu, cnsqu_array) # performing curve fit, and returning parameters
print "poptcnsqu", poptcnsqu

# create the cn array, as it's a large calculation
cn_cnn_squ_array = np.cumsum([cnsqunsqu(n) for n in nn])
popt_cnn_squ, pcov_cnn_squ = curve_fit(expfunc, nn, cn_cnn_squ_array, p0=(1e-6, 1e-6, 1)) # performing curve fit, and returning parameters


class TestCompleteness(unittest.TestCase):

    def test_boundry_conditions(self):
        self.assertAlmostEqual(A(0), 0)
        self.assertAlmostEqual(A(1), 0)

    def test_A_value(self):
        self.assertAlmostEqual(norm(), sqrt(735))

    def test_cn(self):
        self.assertAlmostEqual(cn(1), 0.353298, delta=0.000005)
        self.assertAlmostEqual(cn(2), 0.927407, delta=0.000005)
        self.assertAlmostEqual(cn(3), 0.0130851, delta=0.000005)
        self.assertAlmostEqual(cn(4), 0.115926, delta=0.000005)
        self.assertAlmostEqual(cn(5), 0.00282638, delta=0.000005)
        self.assertAlmostEqual(cn(6), 0.0343484, delta=0.000005)

    def test_cn_goes_to_0(self):
        # popt[1] is b, the negative exponetail factor
        self.assertTrue(popt[1] > 0) # if it is greater than 0, func wil go to
                                     # 0 for large enough x

    def test_cn_squ_goes_to_1(self):
        self.assertTrue(popt[1] > 0) # if it is greater than 0, func wil go to
                                     # 0 for large enough x
        self.assertAlmostEqual(poptcnsqu[2], 1, delta = 0.0001)

    @unittest.expectedFailure
    def test_cnn_squ_goes_to_val(self):
        """fails as the convergance is too sharp to capture in an exponential.
        """
        self.assertAlmostEqual(popt_cnn_squ[2], 3.849, delta = 0.001)

    def test_cnn_squ_goes_to_val_2(self):
        self.assertAlmostEqual(cn_cnn_squ_array[-1], 3.849, delta = 0.01)

def main2():
    xaxis = np.linspace(0, 1, 1000)
    f, axes = plt.subplots(1, 3)

    axes[0].plot(xaxis, A(xaxis))

    print('the norm of A is {}'.format(norm(A)))

    suite = unittest.TestLoader().loadTestsFromTestCase(TestCompleteness)
    unittest.TextTestRunner().run(suite)

    axes[1].plot(nn, cn_array, label="$C_n$")
    print(nn.size, expfunc(nn, *popt).size)
    axes[1].plot(nn, expfunc(nn, *popt), color='green', label='best fit')
    axes[1].legend()

    axes[2].plot(nnsqu, cnsqu_array, color='red', label="$\\Sigma C_n$")
    axes[2].plot(nn, expfunc(nn, *poptcnsqu), color='orange', label='best fit')

    axes[2].plot(nn, cn_cnn_squ_array, color='blue', label="$\\Sigma C_n^2n^2$")
    axes[2].plot(nn, expfunc(nn, *popt_cnn_squ), color='cyan', label='best fit')
    axes[2].legend()
    plt.show()


########## EX3

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

def main3():
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
    main1()
    main2()
    main3()
