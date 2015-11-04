
# PS1EX3
from numpy import sqrt, sin, pi
from scipy.misc import factorial

print(sqrt(17))
print(factorial(18))
print(sin((18 * 2 * pi) / 360))


# PS1EX4
import sys
import numpy as np

G = 9.81


def distance(height, time):
    fallen = 0.5 * G * time**2
    return height - fallen if fallen <= height else 0


if __name__ == '__main__':

    try:
        # also handels non-int input, by ValueError
        given = raw_input("Enter the height and time, space seperated: ")
        height, time = [float(n) for n in given.split(' ')]
    except ValueError:
        print('Not a number. Usage: pset1ex4.py HEIGHT TIME')
        sys.exit()

    final_height = distance(height, time)
    print('The height from the ground at time {0} is {1}'.format(time,
                                                                 final_height))


#PS1EX5
import numpy as np

G = 6.67e-11
M = 5.97e24
R = 6371e3


def height(time):
    orbitRadius = (G * M * time**2 / (4 * np.pi**2)) ** (1. / 3)
    return orbitRadius - R


if __name__ == '__main__':

    time = float(input('Please input a time in seconds: '))
    retString = "A satellite that orbits in time {0} is has altitude {1:.3e}m"
    print(retString.format(time, height(time)))


#PS1EX6
thermo = open('thermodynamics.txt', 'w+')

FirstLaw = """In all cases in which work is produced by the agency of heat,
 a quantity for heat is consumed which is proportional to the work done:
 and conversly, by the expenditure of an equal quantity of work and equal
 quantitiy for heat is produced"""

thermo.write(FirstLaw)

# This is caomment bro
thermo.seek(0)
ret = "The sixth word of this statment of the first law of thermodynamics is {}"
print(ret.format(thermo.read().split()[5]))
thermo.seek(0)
print(thermo.read())
