// Slightly modified version of  Stan Melax's code for 3x3 matrix diagonalization (Thanks Stan!)
// source: http://www.melax.com/diag.html?attredirects=0
/*
Diagonalizing a symmetric 3x3 has various useful applications such as
diagonalizing inertia tensors, fitting OBBs, finding principal axes, etc.
The diagonal entries of the diagonalized matrix are the eigenvalues
and the quaternion represents the eigenvectors in that the rows of the corresponding matrix are the eigenvectors.
Note that would be the matrix's columns for you colum-major people out there.

This code is kept simple so that it should be easy to grab and incorporate into your own 3D math library or game application. Rename vector, matrix and quat types as you see fit. Be aware of any matrix storage (row vs column) and multiplying order conventions that you might be using. The code assumes C-language row-major and D3D conventions for the matrix element ordering (for example v_world=v_local*M). You might have noticed the comments write D=Q*M*Q^T, whereas a column-centric linear algebra textbook would likely write D=Q^T*M*Q instead. The quaternion association with matrices and multiplication is the same ordering that literally everybody uses. To the best of my knowledge, this includes D3DX's quaternion implementation even though its opposite D3D's matrix ordering which would mean: (Qa*Qb).AsMatrix()==Qb.AsMatrix()*Qa.AsMatrix() . Anyways, the function can easily be modified to conform to your preference if its different.

When you call the routine for a matrix M and get a quaternion q whose corresponding matrix is Q and then compute D=Q*M*Q^T you will probably notice that the off diagonal elements of D are not quite zero. The internals of the algorithm are all 32bit float. Changing this to double might improve the result. Even then, the resulting quaternion will be represented with finite precision (32bit xyzw). For the functions main loop, I just hardcoded an iteration limit of 24. No good reason for that number. Hurling dozens of random symmetric matricies at the function, i didn't see it use more than 7 before satisfying one of the exit conditions. Note the random entries were initialized with (float)rand()/(float)rand(). Of what I saw, the offdiagonal elements were always many orders of magnitude smaller than the largest diagonal element and smaller than the smallest diagonal. Further coverage testing in more extreme cases might be useful.

To give credit where credit is due, some guy named Jacobi figured out this diagonalization technique a long time ago. In an iterative fashion, the off diagonal elements are simply "rotated away". The algebra to derive all this is a bit trickier since you have the matrices on both sides (the diagonalizer and its inverse). Numerical recipes 11.1, Jacobi Transformations of a Symmetric Matrix, shows all the derivation including that clever t=s/c substitution which leads to the formula showing how to compute the sin and cos for the next incremental rotation. The Numerical Recipes version makes additional speed optimizations and adds complexity that seems unnecessary for the 3x3 case. I just ignored all that and just used Jacobi's raw idea but incrementally build up a quaternion at each iteration instead of the matrix sequence. (Using the half-angle identity makes it easy to construct the delta quaternion directly.) Consequently the resulting code is fairly simple and should jive with your typical game engine math library. Special thanks to John Schultz for his collaboration on gamedev.net and for trying out the code and comparing results to alternative implementations.
*/

void Diagonalize(const Real (&A)[3][3], Real (&Q)[3][3], Real (&D)[3][3])
{
    // A must be a symmetric matrix.
    // returns Q and D such that
    // Diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
    const int maxsteps=24;  // certainly wont need that many.
    int k0, k1, k2;
    Real o[3], m[3];
    Real q [4] = {0.0,0.0,0.0,1.0};
    Real jr[4];
    Real sqw, sqx, sqy, sqz;
    Real tmp1, tmp2, mq;
    Real AQ[3][3];
    Real thet, sgn, t, c;
    for(int i=0;i < maxsteps;++i)
    {
        // quat to matrix
        sqx      = q[0]*q[0];
        sqy      = q[1]*q[1];
        sqz      = q[2]*q[2];
        sqw      = q[3]*q[3];
        Q[0][0]  = ( sqx - sqy - sqz + sqw);
        Q[1][1]  = (-sqx + sqy - sqz + sqw);
        Q[2][2]  = (-sqx - sqy + sqz + sqw);
        tmp1     = q[0]*q[1];
        tmp2     = q[2]*q[3];
        Q[1][0]  = 2.0 * (tmp1 + tmp2);
        Q[0][1]  = 2.0 * (tmp1 - tmp2);
        tmp1     = q[0]*q[2];
        tmp2     = q[1]*q[3];
        Q[2][0]  = 2.0 * (tmp1 - tmp2);
        Q[0][2]  = 2.0 * (tmp1 + tmp2);
        tmp1     = q[1]*q[2];
        tmp2     = q[0]*q[3];
        Q[2][1]  = 2.0 * (tmp1 + tmp2);
        Q[1][2]  = 2.0 * (tmp1 - tmp2);

        // AQ = A * Q
        AQ[0][0] = Q[0][0]*A[0][0]+Q[1][0]*A[0][1]+Q[2][0]*A[0][2];
        AQ[0][1] = Q[0][1]*A[0][0]+Q[1][1]*A[0][1]+Q[2][1]*A[0][2];
        AQ[0][2] = Q[0][2]*A[0][0]+Q[1][2]*A[0][1]+Q[2][2]*A[0][2];
        AQ[1][0] = Q[0][0]*A[0][1]+Q[1][0]*A[1][1]+Q[2][0]*A[1][2];
        AQ[1][1] = Q[0][1]*A[0][1]+Q[1][1]*A[1][1]+Q[2][1]*A[1][2];
        AQ[1][2] = Q[0][2]*A[0][1]+Q[1][2]*A[1][1]+Q[2][2]*A[1][2];
        AQ[2][0] = Q[0][0]*A[0][2]+Q[1][0]*A[1][2]+Q[2][0]*A[2][2];
        AQ[2][1] = Q[0][1]*A[0][2]+Q[1][1]*A[1][2]+Q[2][1]*A[2][2];
        AQ[2][2] = Q[0][2]*A[0][2]+Q[1][2]*A[1][2]+Q[2][2]*A[2][2];
        // D = Qt * AQ
        D[0][0] = AQ[0][0]*Q[0][0]+AQ[1][0]*Q[1][0]+AQ[2][0]*Q[2][0];
        D[0][1] = AQ[0][0]*Q[0][1]+AQ[1][0]*Q[1][1]+AQ[2][0]*Q[2][1];
        D[0][2] = AQ[0][0]*Q[0][2]+AQ[1][0]*Q[1][2]+AQ[2][0]*Q[2][2];
        D[1][0] = AQ[0][1]*Q[0][0]+AQ[1][1]*Q[1][0]+AQ[2][1]*Q[2][0];
        D[1][1] = AQ[0][1]*Q[0][1]+AQ[1][1]*Q[1][1]+AQ[2][1]*Q[2][1];
        D[1][2] = AQ[0][1]*Q[0][2]+AQ[1][1]*Q[1][2]+AQ[2][1]*Q[2][2];
        D[2][0] = AQ[0][2]*Q[0][0]+AQ[1][2]*Q[1][0]+AQ[2][2]*Q[2][0];
        D[2][1] = AQ[0][2]*Q[0][1]+AQ[1][2]*Q[1][1]+AQ[2][2]*Q[2][1];
        D[2][2] = AQ[0][2]*Q[0][2]+AQ[1][2]*Q[1][2]+AQ[2][2]*Q[2][2];
        o[0]    = D[1][2];
        o[1]    = D[0][2];
        o[2]    = D[0][1];
        m[0]    = fabs(o[0]);
        m[1]    = fabs(o[1]);
        m[2]    = fabs(o[2]);

        k0      = (m[0] > m[1] && m[0] > m[2])?0: (m[1] > m[2])? 1 : 2; // index of largest element of offdiag
        k1      = (k0+1)%3;
        k2      = (k0+2)%3;
        if (o[k0]==0.0)
        {
            break;  // diagonal already
        }
        thet    = (D[k2][k2]-D[k1][k1])/(2.0*o[k0]);
        sgn     = (thet > 0.0)?1.0:-1.0;
        thet   *= sgn; // make it positive
        t       = sgn /(thet +((thet < 1.E6)?sqrt(thet*thet+1.0):thet)) ; // sign(T)/(|T|+sqrt(T^2+1))
        c       = 1.0/sqrt(t*t+1.0); //  c= 1/(t^2+1) , t=s/c
        if(c==1.0)
        {
            break;  // no room for improvement - reached machine precision.
        }
        jr[0 ]  = jr[1] = jr[2] = jr[3] = 0.0;
        jr[k0]  = sgn*sqrt((1.0-c)/2.0);  // using 1/2 angle identity sin(a/2) = sqrt((1-cos(a))/2)
        jr[k0] *= -1.0; // since our quat-to-matrix convention was for v*M instead of M*v
        jr[3 ]  = sqrt(1.0f - jr[k0] * jr[k0]);
        if(jr[3]==1.0)
        {
            break; // reached limits of floating point precision
        }
        q[0]    = (q[3]*jr[0] + q[0]*jr[3] + q[1]*jr[2] - q[2]*jr[1]);
        q[1]    = (q[3]*jr[1] - q[0]*jr[2] + q[1]*jr[3] + q[2]*jr[0]);
        q[2]    = (q[3]*jr[2] + q[0]*jr[1] - q[1]*jr[0] + q[2]*jr[3]);
        q[3]    = (q[3]*jr[3] - q[0]*jr[0] - q[1]*jr[1] - q[2]*jr[2]);
        mq      = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        q[0]   /= mq;
        q[1]   /= mq;
        q[2]   /= mq;
        q[3]   /= mq;
    }
}
