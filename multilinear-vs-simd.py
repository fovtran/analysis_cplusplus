import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

def line_2points(a0, b0, a1, b1):
    L0 = (x - a1) / (a0 - a1)
    L1 = (x - a0) / (a1 - a0)
    return L0 * b0 + L1 * b1

x = sym.Symbol('x')

year = np.arange(1960, 2020, 10)
pop = [179323, 203302, 226542, 249633, 281422, 308746]

eq = 0
for i in range(1, len(year)):
    eq = sym.Piecewise( (line_2points(year[i-1], pop[i-1], year[i], pop[i]), (x >= year[i-1] ) &  (x <= year[i] ) ),
                        (eq, True) )
# sym.plot(eq, (x, year[0], year[-1])) # this also works, but the visualization is much harder to customize

eq_np = sym.lambdify(x, eq)
xs = np.linspace(year[0], year[-1], 200)
plt.plot(xs, eq_np(xs))
plt.plot(year, pop, 'ro')
plt.show()
