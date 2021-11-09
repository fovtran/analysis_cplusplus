from numpy import pi, cos, sin, sqrt, arange
import matplotlib.pyplot as pp

num_pts = 100
indices = arange(0, num_pts, dtype=float) + 0.5

r = sqrt(indices/num_pts)
theta = pi * (1 + 5**0.5) * indices

pp.scatter(r*cos(theta), r*sin(theta))
pp.show()
