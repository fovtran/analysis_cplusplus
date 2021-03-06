import numpy as np
from numpy import linalg as LA

w, v = LA.eig(np.diag((1, 2, 3)))
w,v

(array([1., 2., 3.]),
 array([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]]))


A = [[ 0.35468106,  0.74691448, -0.56242342],
		[ 0.09138513,  0.57095722,  0.81587782],
		[-0.93051067,  0.34077355, -0.13425084]]

w, v = LA.eig(A)
w,v

(array([-1.00000001+0.j        ,  0.89569372+0.44467151j,
         0.89569372-0.44467151j]),
 array([[ 0.53421937+0.j        ,  0.17174654-0.57254516j,
          0.17174654+0.57254516j],
        [-0.4138867 +0.j        ,  0.64369939+0.j        ,
          0.64369939-0.j        ],
        [ 0.73709394+0.j        ,  0.23696882+0.41496029j,
          0.23696882-0.41496029j]]))


 u = v[:,1]
>>> print(u)
[ 0.73595785 -0.38198836 -0.55897311]
>>> lam = w[1]
>>> print(lam)
-2.31662479036
>>> print(np.dot(A,u))
[-1.7049382   0.88492371  1.29493096]
>>> print(lam*u)
[-1.7049382   0.88492371  1.29493096]
