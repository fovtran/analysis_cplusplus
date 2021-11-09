#include <stdio.h>
#include <stdlib.h>
#include <intrin.h>

#pragma intrinsic(__rdtsc)

#define MAX(x,y)    ((x)>(y)?(x):(y))
#define MIN(x,y)    ((x)<(y)?(x):(y))
#define SAT2SI16(x) MAX(MIN((x),32767),-32768)

#define N 100000
unsigned char a[N];
char          b[N];

short sat2short(unsigned char *p, char *q, int n) {
  int i;
  short x = 0;
#ifdef SIMD
	#pragma simd reduction(+:x)
#endif

for (i=0; i<n; i++) {
    x = SAT2SI16(x + p[i]*q[i]);
  }
  return x;
}

int main(int argc, char **argv) {
  short x = 0;
  const __int64 startTime = __rdtsc();

  for (int i=0; i<32767; i++) {
    a[i] = 1;
    b[i] = 1;
  }

#ifdef SIMD
	#pragma simd vectorlength(16)
#endif

for (int i=0; i<32767; i++) {
    if (i >= 16 && i < 32767) {
      b[i] = b[i-16] - 1;
    }
	printf("b[%d] = \n", b[i]);
  }

  x = sat2short(&a[0], &b[0], 32767);

  const __int64 elapsedTime = __rdtsc() - startTime;
  const double elapsedTimeInSecs = (double)elapsedTime / 1000000.0;

  if (x==30847) {
    printf("passed x = %d, time = %lf \n", x, elapsedTimeInSecs);
  }
  else {
    printf("failed x = %d, time = %lf \n", x, elapsedTimeInSecs);
    exit(-1);
  }
}
