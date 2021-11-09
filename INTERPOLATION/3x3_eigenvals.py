import numpy as np
import numpy.linalg as la

eps = 1e-8 # Precision of eigenvalue

def trans(v): # translates vector (v^T)
    v_1 = np.copy(v)
    return v_1.reshape((-1, 1))

def power(A):
    eig = []
    Ac = np.copy(A)
    lamb = 0
    for i in range(3):
        x = np.array([1, 1, 1])
        while True:
            x_1 = Ac.dot(x) # y_n = A*x_(n-1)
            x_norm = la.norm(x_1) 
            x_1 = x_1/x_norm # x_n = y_n/||y_n||
            if(abs(lamb - x_norm) <= eps): # If precision is reached, it returns eigenvalue
                break
            else:
                lamb =dot(x,x_1)
                x = x_1
        eig.append(lamb)

        # Matrix Deflaction: A - Lambda * norm[V]*norm[V]^T
        v = x_1/la.norm(x_1)
        R = v * trans(v)
        R = eig[i]*R
        Ac = Ac - R

    return eig

def main():
    A = np.array([1, 2, 3, 2, 4, 5, 3, 5, -1]).reshape((3, 3))
    print(power(A))



if __name__ == '__main__':
    main()