# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# limits: 0.0 <= t <= 2.0
a = 0.0
b = 2.0

# steps
N = 10

# step-size
h = (b-a)/N

# initial value: y(0.0) = 0.5
IV = (0.0,0.5)

# ordinary differential equation
def f( t, y ):
    return y - t**2 + 1

# make arrays to hold t, and y
t = np.arange( a, b+h, h )
w = np.zeros((N+1,))

# set the initial values
t[0], w[0] = IV

# apply Euler's method
for i in range(1,N+1):
    w[i] = w[i-1] + h * f( t[i-1], w[i-1] )

# exact solution
def y( t ):
    return (t+1.0)**2.0-0.5*np.exp(t)

def euler_mthd( f, a, b, N, IV ):

    # determine step-size
    h = (b-a)/float(N)

    # create mesh
    t = np.arange( a, b+h, h )

    # initialize w
    w = np.zeros((N+1,))

    # set initial values
    t[0], w[0] = IV

    # apply Euler's method
    for i in range(1,N+1):
        w[i] = w[i-1] + h * f( t[i-1], w[i-1] )

    return w

x = np.random.randn(10000)
plt.hist(x, 100)

plt.plot( t, w, label='approximation' )
plt.plot( t, y(t), label='exact' )
plt.title( "Euler's Method Example, N="+str(N) )
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend(loc=4)
plt.grid()
plt.savefig( 'euler_example.png', fmt='PNG', dpi=100 )
