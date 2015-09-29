from mayavi import mlab
import numpy as np

STM = np.loadtxt('stm.dat')

s = mlab.surf(STM)
mlab.show()
