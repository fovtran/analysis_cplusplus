void foo(int *A, int *B, int *restrict C, int n){
  int i;
  int t = 0;

#ifdef PRIVATE
	#pragma simd private(t)
#endif

for (i=0; i<n; i++){
    if (A[i] > 0) {
      t = A[i];
    }
    if (B[i] < 0) {
      t = B[i];
    }
    C[i] = t;
  }
}
