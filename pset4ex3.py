import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from PIL import Image # Pillow

NMR = Image.open('NMR_phantom.png')
R, G, B = NMR.split() # seperate ch.

fig, (top, bottom) = plt.subplots(2, 1) # create the plots

top.imshow(NMR) # give a overview of the image

# create a single channel heatmap of the image
image = bottom.imshow(R)
fig.colorbar(image)

# BONUS NO MARKS.

# find the phantoms
npR = np.array(R.im).reshape(R.size[::-1])
phantom = npR[16:34, 17:35]
conv = signal.convolve2d(npR, phantom) # find the rings by convolving
# ext.imshow(conv)
# http://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
neighborhood_size = 15 # approx size of phantom
threshold = 120000 # the peak height

# find the locations
data_max = filters.maximum_filter(conv, neighborhood_size) # maximum in each location
maxima = (conv == data_max) # true/false array
data_min = filters.minimum_filter(conv, neighborhood_size)
diff = ((data_max - data_min) > threshold)
maxima[diff == 0] = 0

labeled, num_objects = ndimage.label(maxima)
slices = ndimage.find_objects(labeled)
x, y = [], []
for dy,dx in slices:
    x_center = (dx.start + dx.stop - 1)/2 - 8 # center the peaks on the phantoms
    x.append(x_center)
    y_center = (dy.start + dy.stop - 1)/2 - 7
    y.append(y_center)

bottom.plot(x, y, 'ro') # put markers on the centers

for x, y in zip(x, y):
    try:
        # add the means to the points
        bottom.annotate('{:.0f}'.format(npR[y-5:y+5, x-5:x+5].mean()),
                        xy=(x, y),
                        xytext=(x+1, y+1),
                        color='w')
    except:
        # if the x,y was out of bounds, print it
        print x,y

# the cov added a border, crop this
bottom.set_xlim((0, 300))
bottom.set_ylim((140, 0))

# grad = np.gradient(conv)[0]
#
# for row in grad:
#     zero_crossings = np.where(np.diff(np.sign(row)))[0]
#     print zero_crossings
plt.show()
