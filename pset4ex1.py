import matplotlib.pyplot as plt
from matplotlib import gridspec
from numpy import pi

import numpy as np

x = np.linspace(-np.pi, np.pi, 1000) # x-axis is common for all charts
sine = np.sin(x)
cosine = np.cos(x)

# the only sane part of this graph. JUST USE R IF YOU WANT GOOD LOOKIN'
fig, (trigax, expax) = plt.subplots(2, 1)

# plot the data, set colors, keep the line handles for the legend
sineline, = trigax.plot(x, sine, 'b', label='sine')
cosline, = trigax.plot(x, cosine, 'r', label='cosine')

# add the cos, sine legend. loc is the corner, anticlockwise.
trigax.legend(handles=[sineline, cosline], loc=2)

# set the x-limits to reduce the data size. This is a odd graph.
# must be done before moving the splines
# for each axis, we get the current lims and multiply by 1.1
# getters and setters are bad design, that I shall bruteforce
# for s, g in zip([trigax.set_xlim, trigax.set_ylim],
#                 [trigax.get_xlim, trigax.get_ylim]):
#     print(g, s)
#     s([lim*1.1 for lim in g()])

# The multiplier above expanded x too much for ??reasons??.
trigax.set_xlim([-pi*1.1, pi*1.1])
trigax.set_ylim([-1.1, 1.1])

# move axes
trigax.spines['left'].set_position('center')
trigax.spines['right'].set_color('none')
trigax.spines['bottom'].set_position('center')
trigax.spines['top'].set_color('none')
# trigax.spines['left'].set_smart_bounds(True)
#trigax.spines['bottom'].set_smart_bounds(True) # as we have extended the x-axis, stop the plot being cut off
trigax.xaxis.set_ticks_position('bottom')
trigax.yaxis.set_ticks_position('left')

# set the trig x-labels
trigax.set_xticks([-pi, -pi/2, 0, pi/2, pi])

# we mutate the axis lables in place to modify them. Then call update to relayout
# 0 has spaces prefixed to fix layout bug in the original.
a=trigax.get_xticks().tolist()
a[:] = ['$-\pi$', '$-\\frac{\pi}{2}$', '   0', '$+\\frac{\pi}{2}$', '$+\pi$']
trigax.set_xticklabels(a)

# set yticks
trigax.set_yticks([-1, 1])


# cosine label.
trigax.annotate("$\cos{(\\frac{2\pi}{3})}=-\\frac{1}{2}$",
                xy=(2*pi/3, np.cos(2*pi/3,)), xycoords='data', # axis coords are very odd with all the faff above
                xytext=(0.8, -0.8), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0.3"),
                )

# try to get a coversion from data space to axes space. nopeeeeee
# trigtodisplay = trigax.get_xaxis_transform()
# displaytoaxes = trigax.transAxes.inverted()
# cosymin=displaytoaxes.transform(trigtodisplay.transform([0, -0.5]))[1]

cosymin = 0.27 # guess.
trigax.axvline(x=2*pi/3, ymax=0.5,
               ymin=cosymin,
               color='r', linestyle='--')
trigax.plot([2*pi/3], [-0.52], 'ro') # oh dear, more guessing due to lack of transform


# sin label.
trigax.annotate("$\sin{(\\frac{2\pi}{3})}=\\frac{\sqrt{3}}{2}$",
                xy=(2*pi/3, np.sin(2*pi/3,)), xycoords='data', # axis coords are very odd with all the faff above
                xytext=(2.4, 0.8), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0.3"),
                )

# try to get a coversion from data space to axes space. nopeeeeee
# trigtodisplay = trigax.get_xaxis_transform()
# displaytoaxes = trigax.transAxes.inverted()
# cosymin=displaytoaxes.transform(trigtodisplay.transform([0, -0.5]))[1]

sinemin = 0.9 # guess.
trigax.axvline(x=2*pi/3, ymin=0.5,
               ymax=sinemin,
               color='b', linestyle='--')
trigax.plot([2*pi/3], [np.sqrt(3)/2], 'bo') # oh dear, more guessing due to lack of transform



plt.show()
