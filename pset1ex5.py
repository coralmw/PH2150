#! /usr/bin/python3
import numpy as np

G = 6.67e-11
M = 5.97e24
R = 6371e3


def height(time):
    orbitRadius = (G*M*time**2/(4*np.pi**2)) ** (1./3)
    return orbitRadius - R


if __name__ == '__main__':

    time = float(input('Please input a time in seconds: '))
    retString = "A satellite that orbits in time {0} is has altitude {1:.3e}m"
    print(retString.format(time, height(time)))
