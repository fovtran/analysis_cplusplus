import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

x,y, t = sy.symbols('x y t')
x = np.linspace(0, 3, 32)

def F_y(t):
	y= -1*(t**2) + t*3 -1
	return y

N=5
def F_dx(t, x):
    return np.exp(-x*t) / t**N

R1 = integrate.dblquad(F_dx, 0, np.inf,lambda x: 1, lambda x: np.inf)

print (F_y)
print (R1)

sy.plot(x, F_y(x))
#plt.show()
