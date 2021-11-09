import numpy as np
import numpy.linalg as la

eps = 1e-8 # Precision of eigenvalue

def trans(v): # translates vector (v^T)
    v_1 = np.copy(v)
    return v_1.reshape((5, 5))

def power(A):
    eig = []
    Ac = np.copy(A)
    lamb = 0
    for i in range(5):
        x = np.random.uniform(0.0,1.0)
        while True:
            x_1 = Ac.dot(x) # y_n = A*x_(n-1)
            x_norm = la.norm(x_1) 
            x_1 = x_1/x_norm # x_n = y_n/||y_n||
            if(abs(lamb - x_norm) <= eps): # If precision is reached, it returns eigenvalue
                break
            else:
                lamb = x_norm
                x = x_1
        eig.append(lamb)

        # Matrix Deflaction: A - Lambda * norm[V]*norm[V]^T
        v = x_1/la.norm(x_1)
        R = v * trans(v)
        R = eig[i]*R
        Ac = Ac - R

    return eig

def main():
    A = np.array(np.random.uniform(0.0,1.0,25)).reshape((5, 5))
    print(power(A))



if __name__ == '__main__':
    main()
