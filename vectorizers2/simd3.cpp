char foo(char *A, int n){

  int i;
  char x = 0;

#ifdef SIMD
	#pragma simd   // Generates incorrect code
#endif

#ifdef REDUCTION
	#pragma simd reduction(+:x)  // Generates correct code
#endif

#ifdef IVDEP
	#pragma ivdep
#endif

for (i=0; i<n; i++){
    x = x + A[i];
  }
  return x;
}
