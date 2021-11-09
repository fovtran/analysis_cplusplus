# use scipy.integrate.quad(f, a, b) to integrate f(x) from a to b.
#Limits could be infinite, e.g. quad(f, -inf, inf).

import numpy as np
import scipy as sc
from scipy.integrate import quad

np.finfo(dtype=np.float64)
# PC: finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)
# NB: finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)

np.finfo(float).eps
# PC: 2.2204460492503131e-16
# NB: 2.220446049250313e-16

np.real_if_close([2.1 + 4e-14j], tol=1000)
# array([ 2.1])
np.real_if_close([2.1 + 4e-13j], tol=1000)
# array([ 2.1 +4.00000000e-13j])

x2 = lambda x: x**2
res, err = quad(x2, 0, 4)
# (21.333333333333332, 2.3684757858670003e-13)
print(4**3 / 3.)  # analytical result
# 21.333333333333332

f = lambda x: np.cos(x)
res, err = quad(f, 0, np.pi/2)
res
# 0.9999999999999999
err
# 1.1102230246251564e-14

f = lambda x: np.cos(x)
res, err = quad(f, -np.pi, np.pi, full_output=True)

invexp = lambda x: np.exp(-x)
quad(invexp, 0, np.inf)
# (1.0, 5.842605999138044e-11)

# cos(x), sec(x)
