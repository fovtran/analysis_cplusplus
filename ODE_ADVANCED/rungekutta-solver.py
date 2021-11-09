import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt, pi, sin, cos
from scipy.integrate import ode

# use z = [z1, z2] = [u, u']
# and then f = z' = [u', u''] = [z2, -z1+sqrt(z1)]
def f(phi, z):
    return [z[1], -z[0]+sqrt(z[0])]


# initialize the 4th order Runge-Kutta solver
solver = ode(f).set_integrator('dopri5')

# initial value
z0 = [1.49907, 0.]
solver.set_initial_value(z0)

values = 1000
phi = np.linspace(0.0001, 7.*pi, values)
u = np.zeros(values)

for ii in range(values):
    u[ii] = solver.integrate(phi[ii])[0] #z[0]=u

x = 1. / u * cos(phi)
y = 1. / u * sin(phi)

plt.figure()
plt.plot(x,y)
plt.grid()
plt.show()
