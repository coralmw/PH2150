import numpy as np
import matplotlib.pyplot as plt

STM = np.loadtxt('STM.dat')

plt.imshow(STM, cmap=plt.get_cmap('RdPu'))
plt.show()
