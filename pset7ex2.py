import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
The following example traces the movement of a charged particle under the influence of electric and
magnetic fields. You can edit the program to change the components of the fields to see their effect on
the trajectory.
"""

# nupmy gives inaccurate results unless 100x the timepoints taken
# float64 no change
# 0 -> 0. no change
Ei = np.array((0., 2., 0.))
Bi = np.array((0., 0., 4.))
Pi = np.array((0., 0., 0.))
Vi = np.array((20., 10., 2.))

m = 2.0 # Mass of the particle
q = 5.0 # Charge
t = 0.0
T = 10
dt = 0.0001

Peuler = []
E, B, P, V = [vec.copy() for vec in (Ei, Bi, Pi, Vi)]

def ForceAtVelocity(V, E=E, B=B, q=q):
    return q * (E + np.cross(V, B))

while t < T: # trace path until time reaches value e.g. 10
    f = ForceAtVelocity(V)
    a = f / m
    dv = a * dt
    dp = (V+dv)*dt

    V += dv
    P += dp
    Peuler.append(P.copy())
    t += dt

Prk = []
E, B, P, V = [vec.copy() for vec in (Ei, Bi, Pi, Vi)]
t = 0.

while t < T: # trace path until time reaches value e.g. 10
    # first find the first-order update
    f1 = ForceAtVelocity(V)
    a1 = f1 / m
    dv1 = a1 * dt
    dp1 = (V+dv1) * dt # just using this gives the Euler method

    # find the force at the first midpoint.
    f2 = ForceAtVelocity(V+dv1)
    a2 = f2 / m
    dv2 = a2 * (dt/2.)
    dp2 = (V+dv2) * dt/2.

    f3 = ForceAtVelocity(V+dv2)
    a3 = f3 / m
    dv3 = a3 * (dt/2.)
    dp3 = (V+dv3) * dt/2.

    f4 = ForceAtVelocity(V+dv3)
    a4 = f4 / m
    dv4 = a4 * dt
    dp4 = (V+dv4) * dt

    V += dv1/6. + dv2/3. + dv3/3. + dv4/6.
    P += dp1/6. + dp2/3. + dp3/3. + dp4/6.
    Prk.append(P.copy())
    t += dt

fig1=plt.figure()
ax1 = fig1.add_subplot(1,1,1, projection='3d')
ax1.set_title("Path of charged particle under influence of electric and magnetic fields")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
a,b,c = [dim.flatten() for dim in np.hsplit(np.array(Peuler), 3)]
ax1.plot3D(xs=a,ys=b,zs=c, color='blue', label='path')

fig2=plt.figure()
ax2 = fig2.add_subplot(1,1,1, projection='3d')


d,e,f = [dim.flatten() for dim in np.hsplit(np.array(Prk), 3)]
ax2.plot3D(xs=d,ys=e,zs=f, color='red', label='path')

# FLATTEN INS"T MUTATVE SHOULD BE .flat() WHY
# a.flatten()
# b.flatten()
# c.flatten()

ax1.legend(loc='lower left')

plt.show()
