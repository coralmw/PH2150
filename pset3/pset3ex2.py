import numpy as np
import matplotlib.pyplot as plt

stars = np.loadtxt('stars.dat')

plt.plot(stars[::, 0], stars[::, 1], 'bo')
plt.ylim(plt.ylim()[::-1])
plt.xlim(plt.xlim()[::-1])
plt.ylabel('some kind of I mesurement')
plt.xlabel('some kind of temp/color mesurement')
plt.show()
