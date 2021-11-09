from scipy import integrate

def f(x, y):
	return x*y

def bounds_y():
	return [0, 0.5]

def bounds_x(y):
	return [0, 1-2*y]

integrate.nquad(f, [bounds_x, bounds_y])
