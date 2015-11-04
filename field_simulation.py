import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D    
"""
The following example traces the movement of a charged particle under the influence of electric and
magnetic fields. You can edit the program to change the components of the fields to see their effect on
the trajectory.
"""
Ex = 0.0 #  X Component of applied Electric Field
Ey = 2.0
Ez = 0.0
Bx = 0.0 # X Component of applied Magnetic field
By = 0.0
Bz = 4.0
m = 2.0 # Mass of the particle
q = 5.0 # Charge
x = 0.0 # initial position x
y = 0.0 # initial position y
z = 0.0 # initial position z
vx = 20.0 # initial velocity vx
vy = 10.0 # initial velocity vy
vz = 2.0 # initial velocity vz
a = []
b = []
c = []
t = 0.0
dt = 0.01
while t < 10: # trace path until time reaches value e.g. 10
    Fx = q * (Ex + (vy * Bz) - (vz * By) )
    vx = vx + Fx/m * dt # Acceleration = F/m; dv = a.dt
    Fy = q * (Ey - (vx * Bz) + (vz * Bx) )
    vy = vy + Fy/m * dt
    Fz = q * (Ez + (vx * By) - (vy * Bx) )
    vz = vz + Fz/m * dt
    x = x + vx * dt
    y = y + vy * dt
    z = z + vz * dt
    a.append(x)
    b.append(y)
    c.append(z)
    t += dt

fig=plt.figure()
ax = Axes3D(fig)
ax.set_title("Path of charged particle under influence of electric and magnetic fields")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.plot3D(a,b,c, color='blue', label='path')
ax.legend(loc='lower left')

plt.show()