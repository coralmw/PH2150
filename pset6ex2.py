from scipy.integrate import quad
from scipy.special import hermite
from scipy.optimize import curve_fit
from functools import partial
import numpy as np
from numpy import power, sqrt
from pset6ex1 import LengthScale, psi_xnL

import multiprocessing as mp
import matplotlib.pyplot as plt

import unittest


# define the wavefunction
def A(x, L=1):
    '''wavefunc to model.
    x is the xpoint to calculate the value at.
    '''
    return power(x/L, 3) - 11.*np.power(x/L, 2)/7. + 4.*np.power(x/L, 1)/7.

def Anorm(x, L=1):
    return A(x, L)

def norm(wavefunc=A, L=1):
    '''find the norm of a wavefunction.'''
    integral, error = quad(lambda x: A(x)**2, 0, L)
    norm = 1/sqrt(integral)
    return norm

def cn(n, wavefunc=Anorm, sumfunc=psi_xnL, L=1, intlim=50):
    '''for mp.pool. calculates the norm values'''
    return quad(lambda x: psi_xnL(L,n,x)*norm(A)*A(x), 0, 1, limit=intlim)[0]

def cnsqu(n):
    return cn(n, intlim=5000)**2

def cnsqunsqu(n):
    return abs(cn(n))**2*n**2

def expfunc(x, a, b, c):
    """Function for curve fitting."""
    return a*np.exp(-b*x) + c # function to generate data for curve fit

pool = mp.Pool(processes=8)

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


if __name__ == "__main__":
    xaxis = np.linspace(0, 1, 1000)
    f, axes = plt.subplots(1, 3)

    # create the cn array, as it's a large calculation
    nn = np.linspace(1, 50, 50)
    cn_array = pool.map(cn, nn)
    popt, pcov = curve_fit(expfunc, nn, cn_array) # performing curve fit, and returning parameters

    nnsqu = np.linspace(1, 500, 500, dtype=np.int32) # use a much larger array b/c convergance is slower
    cnsqu_array = np.cumsum(pool.map(cnsqu, nnsqu))
    poptcnsqu, pcovcnsqu = curve_fit(expfunc, nnsqu, cnsqu_array) # performing curve fit, and returning parameters
    print "poptcnsqu", poptcnsqu

    # create the cn array, as it's a large calculation
    cn_cnn_squ_array = np.cumsum(pool.map(cnsqunsqu, nn))
    popt_cnn_squ, pcov_cnn_squ = curve_fit(expfunc, nn, cn_cnn_squ_array, p0=(1e-6, 1e-6, 1)) # performing curve fit, and returning parameters


    axes[0].plot(xaxis, A(xaxis))

    print('the norm of A is {}'.format(norm(A)))

    # test lots of properties of the wavefunc
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
