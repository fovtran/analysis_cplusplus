# Evgenii B. Rudnyi, http://MatrixProgramming.com
# Ax = B

import sys
if len(sys.argv) < 3:
	print 'Usage: lu_read matrix rhs'
	print 'Please specify matrix dimensions'
	sys.exit()

import numpy
import scipy.linalg
import scipy.io
import time

A = scipy.io.mmread(sys.argv[1])
# if A is not dense, make it dense
if not isinstance(A, type(numpy.array(1))):
	A = A.toarray()
print 'matrix', A.shape
B = scipy.io.mmread(sys.argv[2])
# if B is not dense, make it dense
if not isinstance(B, type(numpy.array(1))):
	B = B.toarray()
print 'RHS', B.shape

start = time.clock()
lu = scipy.linalg.lu_factor(A)
finish = time.clock()
print 'time for LU is ', finish - start,'s'

start = time.clock()
X = scipy.linalg.lu_solve(lu, B)
finish = time.clock()
scipy.io.mmwrite(sys.argv[2] + '.solve', X)

print 'time for back substitution is ', finish - start,'s'
print 'residual', scipy.linalg.norm(numpy.dot(A, X) - B)//scipy.linalg.norm(A)
