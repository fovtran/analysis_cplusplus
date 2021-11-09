import time
import numpy as np
from math import sqrt
from scipy.linalg import solve_triangular
import math

start_time = time.time()

n=10

################## AAAAA matrix #############################################
A = np.zeros([n, n], dtype=float)  # initialize to f zeros

# ------------------first row
A[0][0] = 6
A[0][1] = -4
A[0][2] = 1
# ------------------second row
A[1][0] = -4
A[1][1] = 6
A[1][2] = -4
A[1][3] = 1
# --------------two last rows-----
# n-2 row
A[- 2][- 1] = -4
A[- 2][- 2] = 6
A[- 2][- 3] = -4
A[- 2][- 4] = 1
# n-1 row
A[- 1][- 1] = 6
A[- 1][- 2] = -4
A[- 1][- 3] = 1

# --------------------------- from second to n-2 row --------------------------#
j = 0
for i in range(2, n - 2):
    if j == (n - 4):
        break
    A[i][j] = 1
    j = j + 1

j = 1
for i in range(2, n - 2):
    if j == (n - 3):
        break
    A[i][j] = -4
    j = j + 1

j = 2
for i in range(2, n - 2):
    if j == (n - 2):
        break
    A[i][j] = 6
    j = j + 1

j = 3
for i in range(2, n - 2):
    if j == (n - 1):
        break
    A[i][j] = -4
    j = j + 1

j = 4
for i in range(2, n - 2):
    if j == (n):
        break
    A[i][j] = 1
    j = j + 1
# -----------------------------end coding of 2nd to n-2 r-------------#
print("\nMatrix A is : \n", A)

####### b matrix ######################################
b = np.zeros(n,float).reshape((n,1))
b[0] = 3
b[1] = -1
#b[len(b) - 1] = 3
#b[len(b) - 2] = -1
b[[0,-1]]=3; b[[1,-2]]=-1

############ init x #####################
x = np.zeros(n,float).reshape((n,1))
#x = [0] * n
#x = np.zeros([n, 1], dtype=float)
print("\n x is ",x)

print("\nMatrix b is \n", b)
#####################################
next_x = 6  # We start the search at x=6
gamma = 0.01  # Step size multiplier
precision = 0.00001  # Desired precision of result
max_iters = 10000  # Maximum number of iterations

# Derivative function
def df(A,b,x):
    return 0.5*(A+A.T.conj())*x - b


    for _i in range(max_iters):
        current_x = next_x
        next_x = current_x - gamma * df(current_x)

        step = next_x - current_x
        if abs(step) <= precision:
            break

    print("Minimum at ", next_x)

myx=df(A,b,x)

print("\n myx is ",myx)
print("--- %s seconds ---" % (time.time() - start_time))
