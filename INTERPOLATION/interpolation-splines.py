# -*- coding: utf-8 -*-
import numpy as np
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d

#x1 = [1., 0.88,  0.67,  0.50,  0.35,  0.27, 0.18,  0.11,  0.08,  0.04,  0.04,  0.02]
#y1 = [0., 13.99, 27.99, 41.98, 55.98, 69.97, 83.97, 97.97, 111.96, 125.96, 139.95, 153.95]
x1 = [2, 1, .8, .1]
y1 = [0, .1, 1, 1]
# Combine lists into list of tuples
points = zip(x1, y1)

# Sort list of tuples by x-value
points = sorted(points, key=lambda point: point[0])

# Split list of tuples into two list of x values any y values
x1, y1 = zip(*points)

new_length = 4
new_x = np.linspace(min(x1), max(x1), new_length)
new_y = sp.interpolate.interp1d(x1, y1, kind='cubic')(new_x)

plot(new_x, new_y, 'bs-', picker=5)
#xlim([-5,5])
#ylim([-5,5])
grid(True)
show()