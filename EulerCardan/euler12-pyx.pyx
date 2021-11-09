# _euler12.pyx
from libc.math cimport sqrt

cdef int factorCount(int n):
    cdef int candidate, isquare, count
    cdef double square
    square = sqrt(n)
    isquare = int(square)
    count = -1 if isquare == square else 0
    for candidate in range(1, isquare + 1):
        if not n % candidate: count += 2
    return count

cpdef main():
    cdef int triangle = 1, index = 1
    while factorCount(triangle) < 1001:
        index += 1
        triangle += index
    print triangle

# euler12-cython.py
import _euler12
_euler12.main()

# setup.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("_euler12", ["_euler12.pyx"])]

setup(
  name = 'Euler12-Cython',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
