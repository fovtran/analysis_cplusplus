from math import cos, exp, pi
from scipy.integrate import quad

# function we want to integrate
def f(x):
    return exp(cos(-2 * x * pi)) + 3.2

# call quad to integrate f from -2 to 2
res, err = quad(f, -2, 2)

print("The numerical result is {:f} (+-{:g})"
    .format(res, err))


def f(x):
  return np.sin(x)

from scipy.integrate import quad

res,err=quad(f, -2,2)

err #  3.1278966234842506e-14
res #  0.0

res,err=quad(f, -np.pi,np.pi)

err #  4.3998892617846e-14
res #  0.0

res,err=quad(f, -np.pi,np.pi)

res #  1.743934249004316e-16
err #  4.471737093238828e-14
