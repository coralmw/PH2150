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
    expx, expxerr = quad(lambda x: psi_xn(n, x)*x*psi_xn(n, x), 0, LengthScale, limit=50*n)
    expx2, expx2err = quad(lambda x: psi_xn(n, x)*x*x*psi_xn(n, x), 0, LengthScale, limit=50*n)
    return np.sqrt(expx2 - expx**2)


def main():
    xaxis = np.linspace(-LengthScale, LengthScale, 1000)

    fig, (sep, together) = plt.subplots(nrows=2, ncols=1)

    sep.plot(xaxis, psi_xn(3, xaxis), 'r', label="n=3")
    sep.plot(xaxis, psi_xn(4, xaxis), 'g', label="n=4")
    sep.legend()

    product34 = psi_xn(3, xaxis)*psi_xn(4, xaxis) # show that the product goes to 0
    together.plot(xaxis, product34, label='product')
    together.axhline(product34.mean(), color='r', label='mean')
    together.legend()
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
    for n in [5, 7, 9]: # lots of diffrent QN's
        expx, expxerr = quad(lambda x: psi_xn(n, x)*x*psi_xn(n, x), 0, LengthScale)
        print("for psi{}, expx={}, L={}".format(n,expx,LengthScale))

    for n in [4, 7, 100]:
        # test the uncirt is equal to the given formula
        expx, expxerr = quad(lambda x: psi_xn(n, x)*x*psi_xn(n, x), 0, LengthScale)
        expx2, expx2err = quad(lambda x: psi_xn(n, x)*x*x*psi_xn(n, x), 0, LengthScale)
        print("for psi{}, expx={}, x^2={}, L={}".format(n,expx,expx2,LengthScale))
        print("uncirt x={}".format(np.sqrt(expx2 - expx**2)))

    n = np.logspace(1, 5, 100, dtype=np.int32)

    # test the convergance of dx with more terms in the sum
    pool = mp.Pool(processes=8)
    vals = pool.map(dx, n) # map to use all 8 cores
    plt.semilogx(n, vals, label='delta x')
    plt.axhline(LengthScale/np.sqrt(12), label='$\\frac{L}{\sqrt{12}}$', color='r')

    #plt.semilogx(n, abs(LengthScale/np.sqrt(12) - vals), label='error')
    plt.legend()
    plt.xlabel('terms in sum')
    plt.title('convergance')

    plt.show()

if __name__ == "__main__":
    main()
