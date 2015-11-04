import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from sys import float_info

def lin(x, a):
    return -a*x

# Make empty lists
tlist = []
xlist = []
vlist = []
#The initial values and constant are set next.
# Set inital values and constants
t = 0.0
dt = 0.001   # time step
x = 0.
v = 5.
tf = 20.0

#The initial values are appended to the lists.
# Append inital values to the lists
tlist.append(t)
xlist.append(x)
vlist.append(v)
#Inside of a loop, equations (3) and (4) are used to find values of the derivative and the function (in that order) at the next time.
#The type of loop used depends on how you want to determine when the program will stop.
#The new values are appended to the lists.
while t<tf:
    # Calculate new values
    a = -4*x - 0.2*v      # Calculate a using the current x & v
    v = v + a*dt  # Use the current a to update v
    x = x + v*dt  # Use the new v to update x
    t = t + dt    # Advance the time by a step dt
    # Append new values to lists
    tlist.append(t)
    xlist.append(x)
    vlist.append(v)
#The results are plotted.
# Plot the position vs. time
plt.figure()
plt.plot(tlist, xlist, ls='-', c='r')
plt.xlabel('time')
plt.ylabel('position')
plt.show()
