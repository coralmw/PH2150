'''Module for doing RK solving of a electrons motion in electric
and magnetic spaces.

CC Thomas Parks 2015
thomasparks@outlook.com
thomas.parks.2013@live.rhul.ac.uk

Lisenced under MIT availabe from https://opensource.org/licenses/MIT
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def VectorListToPLOT3D(vecs):
    '''Returns a form suitable for plot3d.'''
    return dict(zip(
                     ['xs','ys','zs'],
                     [dim.flatten() for dim in np.hsplit(np.array(vecs), 3)]
                    ))


class Particle(object):

    def __init__(self, V, P, q, m):
        if type(V) != np.ndarray:
            print V, type(V)
            raise ValueError('V is the wrong kind or shape')

        if type(P) != np.ndarray:
            print P, type(P)
            raise ValueError('P is the wrong kind or shape')

        if type(q) not in (float, int):
            print type(q)
            print q
            print 'HERRRRRRE'

        assert(isinstance(q, (int, long, float, complex)))
        assert(isinstance(m, (int, long, float, complex)))

        self.V = V
        self.P = P
        self.m = m
        self.q = q

    def copy(self):
        return Particle(self.V.copy(), self.P.copy(), self.m, self.q)

    def __repr__(self):
        return "Position:{}, V:{}".format(self.P, self.V)


def ForceAtVelocity(E, B, particle):
    '''find the force at a given vector location.
    takes vectors, E,B and a scalr q.
    returns vector of the force.'''
    return particle.q * (E + np.cross(particle.V, B)) # Faraday's law


def Euler(E, B, particle, Tmax=10, dt=0.001):
    assert(type(E) == np.ndarray)
    assert(type(B) == np.ndarray)
    Peuler = []
    t = 0.

    while t < Tmax: # trace path until time reaches value e.g. 10
        f = ForceAtVelocity(E, B, particle)
        a = f / particle.m
        dv = a * dt
        dp = (particle.V+dv)*dt # do a iteration using eulers method

        particle.V += dv
        particle.P += dp
        Peuler.append(particle.P.copy())
        t += dt

    return Peuler


def RK(E, B, particle, Tmax=10, dt=0.001):
    Prk = []
    t = 0.

    while t < Tmax: # trace path until time reaches value e.g. 10
        # first find the first-order update
        f1 = ForceAtVelocity(E, B, particle)
        a1 = f1 / particle.m
        dv1 = a1 * dt
        dp1 = (particle.V+dv1) * dt # just using this gives the Euler method

        # find the force at the first midpoint.
        midParticle1 = particle.copy()
        midParticle1.V += (dv1/2)
        f2 = ForceAtVelocity(E, B, midParticle1)
        a2 = f2 / particle.m
        dv2 = a2 * dt
        dp2 = (particle.V+dv2) * dt

        # second midpoint and so on
        midParticle2 = particle.copy()
        midParticle2.V += (dv2/2)
        f3 = ForceAtVelocity(E, B, midParticle1)
        a3 = f3 / particle.m
        dv3 = a3 * dt
        dp3 = (particle.V+dv3) * dt

        midParticle3 = particle.copy()
        midParticle3.V += dv3
        f4 = ForceAtVelocity(E, B, midParticle3)
        a4 = f4 / particle.m
        dv4 = a4 * dt
        dp4 = (particle.V+dv4) * dt

        particle.V += dv1/6. + dv2/3. + dv3/3. + dv4/6. # use RK for both V and P, for MOR ACCUR.
        particle.P += dp1/6. + dp2/3. + dp3/3. + dp4/6.
        Prk.append(particle.P.copy())
        t += dt

    return Prk




if __name__ == '__main__':

    E = np.array((0., 2., 0.)) # all vectors for neatness
    B = np.array((0., 0., 4.))
    P = np.array((0., 0., 0.))
    V = np.array((20., 10., 2.))

    m = 2.0 # Mass of the particle
    q = 5.0 # Charge
    t = 0.0
    T = 6
    dt = 0.001

    ParticleE = Particle(V, P, m, q)
    ParticleRK = ParticleE.copy()

    Peuler, Prk = Euler(E, B, ParticleE, Tmax=T, dt=dt), RK(E, B, ParticleRK, Tmax=T, dt=dt)
    print Peuler

    fig1=plt.figure()
    ax1 = fig1.add_subplot(1,1,1, projection='3d')
    ax1.set_title("Path of charged particle under influence of electric and magnetic fields")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.plot3D(color='blue', label='Euler', **VectorListToPLOT3D(Peuler))
    ax1.plot3D(color='red', label='RK', **VectorListToPLOT3D(Prk))

    plt.show()
