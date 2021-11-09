# Evgenii B. Rudnyi, http://MatrixProgramming.com
# Ax = B

import sys
if len(sys.argv) < 2:
	print 'Usage: lu dim rhs'
	print 'Please specify matrix dimensions'
	sys.exit()

import numpy
import scipy.linalg
import time

dim = int(sys.argv[1])
if len(sys.argv) < 3:
	rhs = 1
else:
	rhs = int(sys.argv[2])

A = numpy.matrix(numpy.random.rand(dim,dim))
B = numpy.matrix(numpy.random.rand(dim,rhs))

start = time.clock()
lu = scipy.linalg.lu_factor(A)
finish = time.clock()
print 'time for LU is ', finish - start,'s'

start = time.clock()
X = scipy.linalg.lu_solve(lu, B)
finish = time.clock()
print 'time for back substitution is ', finish - start,'s'
print 'residual', scipy.linalg.norm(numpy.dot(A, X) - B)/scipy.linalg.norm(A)
