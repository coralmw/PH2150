## ex1
from sys import exit

import numpy as np

PS3a = np.zeros((4, 4)) # 4x4 array of 0
PS3b = np.linspace(1, 16, 16).reshape((4, 4)) # 1-16, in a 4x4 array
PS3c = np.random.randn(4, 4)

# vstack to put the seperate arrays on top
# linspace takes the args
# the list is a list of the args to linspace for each row. the final array is laid out like the linspace args.
PS3d = np.vstack([np.linspace(**args) for args in [{'start':1, 'stop':17, 'num':4},
                                                   {'start':1, 'stop':2, 'num':4},
                                                   {'start':100, 'stop':200, 'num':4},
                                                   {'start':100, 'stop':200, 'num':4, 'dtype':np.int16}]])

PS3e = PS3a + PS3b # elem-wise addition
print(PS3e[3, 2]) # the 4th, 3rd elem

PS3f = np.power(PS3b, 2) # ^2
PS3e = np.dot(PS3b, PS3b) - PS3f # as said

PS3g = np.power(np.sin(PS3c), 2) # power of the sin of the arr
PS3c.transpose()
PS3c.diagonal() # 1d array

np.concatenate((PS3a, PS3b), axis=1)
np.concatenate((PS3a, PS3c))

## ex2
import numpy as np
import matplotlib.pyplot as plt

stars = np.loadtxt('stars.dat')

plt.plot(stars[::, 0], stars[::, 1], 'bo') # take each point, and take the x,y axes
plt.ylim(plt.ylim()[::-1])
plt.xlim(plt.xlim()[::-1])
plt.ylabel('some kind of I mesurement')
plt.xlabel('some kind of temp/color mesurement')
plt.show()

## ex3

import numpy as np
import matplotlib.pyplot as plt

STM = np.loadtxt('STM.dat')

plt.imshow(STM, cmap=plt.get_cmap('RdPu')) # nicer colourmap
plt.show() # show the image
