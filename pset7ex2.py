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

def dP(P, V, E=E, B=E, q=q, dt=dt):
    F = q * (E + np.cross(V, B))
    V += F/m * dt # Acceleration = F/m; dv = a.dt
    return V*dt

while t < T: # trace path until time reaches value e.g. 10
    P = dP(P, V)
    Peuler.append(P.copy())
    t += dt

Prk = []
E, B, P, V = [vec.copy() for vec in (Ei, Bi, Pi, Vi)]
t = 0.

while t < T: # trace path until time reaches value e.g. 10
    k1 = dP(P, V)
    k2 = dP(P + k1/2, V)
    k3 = dP(P + k2/2, V)
    k4 = dP(P + k3/2, V)
    P = P + k1/6. + k2/3. + k3/3. + k4/6.
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
