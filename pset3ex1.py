import numpy as np

PS3a = np.zeros((4, 4)) 
PS3b = np.linspace(1, 16, 16).reshape((4, 4))
PS3c = np.random.randn(4, 4)

PS3d = np.vstack([np.linspace(**args) for args in [{'start':1, 'stop':17, 'num':4},
                                                   {'start':1, 'stop':2, 'num':4},
                                                   {'start':100, 'stop':200, 'num':4},
                                                   {'start':100, 'stop':200, 'num':4, 'dtype':np.int16}]])

PS3e = PS3a + PS3b
print(PS3e[3, 2])

PS3f = np.power(PS3b, 2)
PS3e = np.dot(PS3b, PS3b) - PS3f

PS3g = np.power(np.sin(PS3c), 2)
PS3c.transpose()
PS3c.diagonal()

np.concatenate((PS3a, PS3b), axis=1)
np.concatenate((PS3a, PS3c))
