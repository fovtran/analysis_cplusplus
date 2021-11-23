#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double stoptime(void) {
   struct timeval t;
   gettimeofday(&t,NULL);
   return (double) t.tv_sec + t.tv_usec/1000000.0;
}

double addmul(double add, double mul, int ops){
   // Need to initialise differently otherwise compiler might optimise away
   double sum1=0.1, sum2=-0.1, sum3=0.2, sum4=-0.2, sum5=0.0;
   double mul1=1.0, mul2= 1.1, mul3=1.2, mul4= 1.3, mul5=1.4;
   int loops=ops/10;          // We have 10 floating point operations inside the loop
   double expected = 5.0*add*loops + (sum1+sum2+sum3+sum4+sum5)
               + pow(mul,loops)*(mul1+mul2+mul3+mul4+mul5);

   for (int i=0; i<loops; i++) {
      mul1*=mul; mul2*=mul; mul3*=mul; mul4*=mul; mul5*=mul;
      sum1+=add; sum2+=add; sum3+=add; sum4+=add; sum5+=add;
   }
   return  sum1+sum2+sum3+sum4+sum5+mul1+mul2+mul3+mul4+mul5 - expected;
}

int main(int argc, char** argv) {
   if (argc != 2) {
      printf("usage: %s <num>\n", argv[0]);
      printf("number of operations: <num> millions\n");
      exit(EXIT_FAILURE);
   }
   int n = atoi(argv[1]) * 1000000;
   if (n<=0)
       n=1000;

   double x = M_PI;
   double y = 1.0 + 1e-8;
   double t = stoptime();
   x = addmul(x, y, n);
   t = stoptime() - t;
   printf("addmul:\t %.3f s, %.3f Gflops, res=%f\n", t, (double)n/t/1e9, x);
   return EXIT_SUCCESS;
}
Compiled with:

g++ -O2 -march=native addmul.cpp ; ./a.out 1000
produces the following output on an Intel Core i5-750, 2.66 GHz:

addmul:  0.270 s, 3.707 Gflops, res=1.326463
That is, just about 1.4 flops per cycle. Looking at the assembler code with g++ -S -O2 -march=native -masm=intel addmul.cpp the main loop seems kind of optimal to me.
  for (int i=0; i<loops/3; i++) {
       mul1*=mul; mul2*=mul; mul3*=mul;
       sum1+=add; sum2+=add; sum3+=add;
       mul4*=mul; mul5*=mul; mul1*=mul;
       sum4+=add; sum5+=add; sum1+=add;

       mul2*=mul; mul3*=mul; mul4*=mul;
       sum2+=add; sum3+=add; sum4+=add;
       mul5*=mul; mul1*=mul; mul2*=mul;
       sum5+=add; sum1+=add; sum2+=add;

       mul3*=mul; mul4*=mul; mul5*=mul;
       sum3+=add; sum4+=add; sum5+=add;
   }
inc    eax