import matplotlib.pyplot as plt
from matplotlib import gridspec
from numpy import pi

import numpy as np

x = np.linspace(-np.pi, np.pi, 1000) # x-axis is common for all charts
sine = np.sin(x)
cosine = np.cos(x)

# get axis objects to plot the data to
fig, (trigax, expax) = plt.subplots(2, 1)

# set the title
fig.suptitle("Figure showing two subplots, with points labeled using LaTeX text.", fontsize='small')

# plot the data, set colors, keep the line handles for the legend
sineline, = trigax.plot(x, sine, 'r', linewidth=2, label='sine')
cosline, = trigax.plot(x, cosine, 'b', linewidth=2, label='cosine')

# add the cos, sine legend. loc is the corner, anticlockwise.
trigax.legend(handles=[sineline, cosline], loc=2, fontsize='x-small')

# set the x-limits to reduce the data size.
# must be done before moving the splines
# for each axis, we get the current lims and multiply by 1.1
# getters and setters are bad design, that I shall bruteforce
# for s, g in zip([trigax.set_xlim, trigax.set_ylim],
#                 [trigax.get_xlim, trigax.get_ylim]):
#     print(g, s)
#     s([lim*1.1 for lim in g()])

# The multiplier above expanded x too much for an unknown reason.
trigax.set_xlim([-pi*1.1, pi*1.1])
trigax.set_ylim([-1.1, 1.1])

# move axes
trigax.spines['left'].set_position('center')
trigax.spines['right'].set_color('none')
trigax.spines['bottom'].set_position('center')
trigax.spines['top'].set_color('none')
# trigax.spines['left'].set_smart_bounds(True) # no need for smart bounds as we set the ax lims manually
#trigax.spines['bottom'].set_smart_bounds(True) # as we have extended the x-axis, stop the plot being cut off
trigax.xaxis.set_ticks_position('bottom') # need to set ticks pos so that they do not show up on the hidden axes
trigax.yaxis.set_ticks_position('left')

# set the trig x-labels
trigax.set_xticks([-pi, -pi/2, 0, pi/2, pi])

# we mutate the axis lables in place to modify them. Then call update to relayout
# 0 has spaces prefixed to fix layout bug in the original.
trigax.set_xticklabels(['$-\pi$', '$-\\frac{\pi}{2}$', '   0', '$+\\frac{\pi}{2}$', '$+\pi$'])

# set yticks on trigax
trigax.set_yticks([-1, 1])

# cosine label.
trigax.annotate("$\cos{(\\frac{2\pi}{3})}=-\\frac{1}{2}$",
                xy=(2*pi/3, np.cos(2*pi/3,)), xycoords='data', # axis coords are very odd with all the faff above
                xytext=(0.8, -0.8), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0.3"),
                )

# try to get a conversion from data space to axes space. nopeeeeee
# trigtodisplay = trigax.get_xaxis_transform()
# displaytoaxes = trigax.transAxes.inverted()
# cosymin=displaytoaxes.transform(trigtodisplay.transform([0, -0.5]))[1]

cosymin = 0.27 # guess.
trigax.axvline(x=2*pi/3, ymax=0.5,
               ymin=cosymin,
               color='b', linestyle='--')
trigax.plot([2*pi/3], [-0.52], 'bo') # oh dear, more guessing due to lack of transform


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
               color='r', linestyle='--')
trigax.plot([2*pi/3], [np.sqrt(3)/2], 'ro') # oh dear, more guessing due to lack of transform

exp = np.exp(x)
negexp = np.exp(-x)

expax.plot([0], [1], 'bo') # do this plot here to have the blob behind the lines.
expline, = expax.plot(x, exp, 'g', linewidth=2, label='exp')
negexpline, = expax.plot(x, negexp, 'c', linewidth=2, label='- exp')
# clockwise label loc = bottom left
expax.legend(handles=[expline, negexpline], loc=3, fontsize='x-small')

# the axis limits expantion is inherited from above.
# ... I bet this wasn't intentional on the orj plot authors part


# x-limits the same as above. I know you can share.
expax.set_xlim([-pi*1.1, pi*1.1])

# hiding axes again
expax.spines['left'].set_position('center')
expax.spines['right'].set_color('none')
expax.spines['top'].set_color('none')
expax.xaxis.set_ticks_position('bottom')
expax.yaxis.set_ticks_position('left')

expax.set_xticks([-pi, -pi/2, 0, pi/2, pi])
expax.set_yticks([0, 10, np.exp(pi)])

# we mutate the axis lables in place to modify them. Then call update to relayout
# 0 has spaces prefixed to fix layout bug in the original.
expax.set_yticklabels(['0', '10', '$\exp{(\pi)}$'])
expax.set_xticklabels(['$-\pi$', '$-\\frac{\pi}{2}$', '   0', '$+\\frac{\pi}{2}$', '$+\pi$'])


expax.annotate("$\exp{(0)}=1$",
               xy=(0, 1), xycoords='data', # axis coords are very odd with all the faff above
               xytext=(0.3, 4), textcoords='data',
               arrowprops=dict(arrowstyle="->",
                           connectionstyle="arc3, rad=0.3"),
               )
expax.plot([0], [1], 'bo') # oh dear, more guessing due to lack of transform


plt.savefig('pset4ex1.png')
