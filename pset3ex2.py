import numpy as np
import matplotlib.pyplot as plt

stars = np.loadtxt('stars.dat')

plt.plot(stars[::, 1], stars[::, 0], 'bo')
plt.ylabel('some kind of I mesurement')
plt.xlabel('some kind of temp/color mesurement')
plt.show()
