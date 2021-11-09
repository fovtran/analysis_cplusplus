import numpy as np

# hardware vectorize uses integers only not float
@np.vectorize
def fun(r):
	if r <= 0.0:
		return 0.0
	else:
		return min(2.0/(1.0 + r), 2.0*r/(1.0 + r))

def fun2(r):
  ans = np.zeros_like(r)
  ans[r > 0.0] = np.min(2.0/(1.0 + r), 2.0*r/(1.0 + r))
  return ans

def fun3(r):
  return np.piecewise(r
                     , [r <= 0.0, r > 0.0]
                     , [0.0, lambda x: np.min(2.0/(1.0 + x), 2.0*x/(1.0 + x))]
                     )

def fun4(r):
  np.clip(r, 0.0, r, out=r)
  return 2.0 * np.minimum(1.0, r) / (1.0 + r)

@np.vectorize
def Ham1(t):
    d=np.array([[np.cos(t),np.sqrt(t)],[0,1]],dtype=np.complex128)
    return d

def Ham2(t):
    d=np.array([[np.cos(t),np.sqrt(t)],[0,1]],dtype=np.complex128)
    return d

HamVec1 = np.vectorize(Ham1, otypes=[np.ndarray])
HamVec2 = np.vectorize(Ham2, otypes=[np.ndarray])

x=np.array([1,2,3])
y = np.linspace(0,10,10000)
# print( type(x[0][0][0]) )
print (x*np.complex('3+2j'))

print( HamVec1(x) )
print( HamVec2(x) )
