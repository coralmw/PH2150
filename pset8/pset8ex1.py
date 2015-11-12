import numpy as np
import matplotlib.pyplot as plt

class VelocityProfile(object):

    def __init__(self, R, beta, mu0, n):
        self.R = float(R)
        self.beta = float(beta)
        self.mu0 = float(mu0)
        self.n = float(n)

    def _v_loc(self, r):
        '''calculates the velocity at a distance r from the center of the pipe.
        private function.
        '''
        if type(r) == np.ndarray and (r > self.R).any():
            raise ValueError('r must be within the pipe.')

        prefactor = (self.beta/(2*self.mu0))**(1./self.n)
        middle = self.n/(self.n+1)
        rdep = ( self.R**(1+1/self.n) - abs(r)**(1+1/self.n) )

        return prefactor*middle*rdep

    def vProfile(self, res=1000):
        rarr = np.linspace(-self.R, self.R, res)
        varr = self._v_loc(rarr)
        return rarr, varr


smallPipe = VelocityProfile(1, 0.06, 0.02, 0.1)
largePipe = VelocityProfile(2, 0.03, 0.02, 0.1)
plt.plot(*smallPipe.vProfile(), label='small pipe')
plt.plot(*largePipe.vProfile(), label='large pipe')
plt.legend()
plt.xlabel('r')
plt.ylabel('velocity')

plt.show()
