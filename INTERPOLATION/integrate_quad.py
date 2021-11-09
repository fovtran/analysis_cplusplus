import numpy as np
from scipy import integrate

N=5
def f(t, x):
    return np.exp(-x*t) / t**N

R1=integrate.dblquad(f,0, np.inf,lambda x: 1, lambda x: np.inf)
print (R1)