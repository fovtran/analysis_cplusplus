int seidel_p(double *A, double *b, double *x, int n) {
    double *dx;
    dx = (double*) calloc(n,sizeof(double));
    int epsilon = 1.0e-4;
    int i,j,k,id;
    double dxi;
    double sum;
    for(int i =0; i<n ; i++)
    {
        x[i] = 0;
    }

    int maxit = 2*n*n;

    for(k=0; k<maxit; k++) {
      for(i=0; i<n; i++) {
         dx[i] = b[i];
         #pragma omp parallel shared(A,x)
         {
            for(j=0; j<n; j++) {
                if(i!=j) {
                    dxi += A[i*n+j]*x[j];
                }
            }

            #pragma omp critical
               dx[i] -= dxi;
         }

         dx[i] /= A[i*n +i];
         x[i] = dx[i];
         sum += ( (dx[i] >= 0.0) ? dx[i] : -dx[i]);
         if(sum<= epsilon) break;
      }
   }

   for( int i = 0 ; i<n ; i++ ) { printf("\t\t%f\n", dx[i]); }
   free(dx);
}
