from mayavi import mlab
import numpy as np

# load some data from a txt file
STM = np.loadtxt('stm.dat')

# display the surface.
s = mlab.surf(STM)
#mlab.show()
