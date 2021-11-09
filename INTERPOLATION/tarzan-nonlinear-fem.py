import numpy as np
    import matplotlib.pyplot as plt
        
    def stiffassembly(M):
        x = np.linspace(0,1,M+1)
        diag = np.zeros(M-1) #x_1,...,x_M-1 (M-1)
        subd = np.zeros(M-2)
        supr = np.zeros(M-2)
        h = np.diff(x)
        for i in range(1,M):
            diag[i-1] = 1/h[i-1] +1/h[i]

        for k in range(1,M-1):
            supr[k-1] = -1/h[k]
            subd[k-1] = -1/h[k]

        A = np.diag(subd,-1) + np.diag(diag,0) + np.diag(supr,+1)
        return A


    def massmatrix(N):
        x = np.linspace(0,1,N+1)
        diag = np.zeros(N-1) #x_1,...,x_M-1 (M-1)
        subd = np.zeros(N-2)
        supr = np.zeros(N-2)
        h = np.diff(x)
        for i in range(1,N):
            diag[i-1] = (h[i-1] + h[i])/3

        for k in range(1,N-1):
            supr[k-1] = h[k]/6
            subd[k-1] = h[k-1]/6

        M = np.diag(subd,-1) + np.diag(diag,0) + np.diag(supr,+1)
        return M


    def inidata(x):
        return np.sin(np.pi*x)



    a = lambda w: (1. * w) ** 2


    M = 50
    x = np.linspace(0,1,M+1)
    delta = 0.001
    odx = 1.0/delta
    tol = 1e-14
    uprev = inidata(x[1:-1])
    ts = 1000 #integration up to t=1.0
    for n in range(ts):
        print('iteration',str(n))
        u = uprev.copy()
        uold = u.copy() + 1
        it = 0
        while (np.linalg.norm(u-uold)>tol):
            uold=u.copy()
            u = np.linalg.solve(odx*massmatrix(M) + np.diag(a(u))@stiffassembly(M), odx*massmatrix(M)@uprev)
            errnrm = np.linalg.norm(u-uold)
            print(errnrm)
        uprev = u.copy()


    plt.plot(x,np.r_[0,u,0],'g-o',)
