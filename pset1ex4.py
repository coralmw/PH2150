#! /usr/bin/python3
import sys
import numpy as np

G = 9.81


def distance(height, time):
    fallen = 0.5 * G * time**2
    return height - fallen if fallen <= height else 0


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: pset1ex4.py HEIGHT TIME')
        sys.exit()

    try:
        # also handels non-int input
        height, time = [float(n) for n in input(
            'Enter the height and time, space seperated: ').split()]
    except ValueError:
        print('Not a number. Usage: pset1ex4.py HEIGHT TIME')
        sys.exit()

    final_height = distance(height, time)
    print('The height from the ground at time {0} is {1}'.format(time,
                                                                 final_height))
