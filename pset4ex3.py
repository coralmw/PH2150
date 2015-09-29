import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from PIL import Image

NMR = Image.open('NMR_phantom.png')
R, G, B = NMR.split()

fig, (top, bottom) = plt.subplots(2, 1) # create the plots

top.imshow(NMR) # give a overview of the image

# create a single channel heatmap of the image
image = bottom.imshow(R)
fig.colorbar(image)

# find the phantoms
npR = np.array(R.im).reshape(R.size[::-1])
phantom = npR[16:34, 17:35]
conv = signal.convolve2d(npR, phantom) # find the rings by convolving

# http://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
neighborhood_size = 15
threshold = 120000
data_max = filters.maximum_filter(conv, neighborhood_size)
maxima = (conv == data_max)
data_min = filters.minimum_filter(conv, neighborhood_size)
diff = ((data_max - data_min) > threshold)
maxima[diff == 0] = 0

labeled, num_objects = ndimage.label(maxima)
slices = ndimage.find_objects(labeled)
x, y = [], []
for dy,dx in slices:
    x_center = (dx.start + dx.stop - 1)/2 - 8
    x.append(x_center)
    y_center = (dy.start + dy.stop - 1)/2 - 7
    y.append(y_center)

bottom.plot(x, y, 'ro') # put markers on the centers

for x, y in zip(x, y):
    try:

        bottom.annotate(npR[y, x], xy=(x, y), xytext=(x+1, y+1))
    except:
        print x,y


# grad = np.gradient(conv)[0]
#
# for row in grad:
#     zero_crossings = np.where(np.diff(np.sign(row)))[0]
#     print zero_crossings
plt.show()
