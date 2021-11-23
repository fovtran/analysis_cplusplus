
inline float TestSqrtFunction( float in );

void TestFunc()
{
  #define ARRAYSIZE 4096
  #define NUMITERS 16386
  float flIn[ ARRAYSIZE ]; // filled with random numbers ( 0 .. 2^22 )
  float flOut [ ARRAYSIZE ]; // filled with 0 to force fetch into L1 cache

  cyclecounter.Start();
  for ( int i = 0 ; i < NUMITERS ; ++i )
    for ( int j = 0 ; j < ARRAYSIZE ; ++j )
    {
       flOut[j] = TestSqrtFunction( flIn[j] );
       // unrolling this loop makes no difference -- I tested it.
    }
  cyclecounter.Stop();
  printf( "%d loops over %d floats took %.3f milliseconds",
          NUMITERS, ARRAYSIZE, cyclecounter.Milliseconds() );
}


// I've tried this with a few different bodies for the TestSqrtFunction, and I've got some timings that are really scratching my head. The worst of all by far was using the native sqrt() function and letting the "smart" compiler "optimize". At 24ns/float, using the x87 FPU this was pathetically bad:

inline float TestSqrtFunction( float in )
{  return sqrt(in); }

// The next thing I tried was using an intrinsic to force the compiler to use SSE's scalar sqrt opcode:

inline void SSESqrt( float * restrict pOut, float * restrict pIn )
{
   _mm_store_ss( pOut, _mm_sqrt_ss( _mm_load_ss( pIn ) ) );
   // compiles to movss, sqrtss, movss
}

// This was better, at 11.9ns/float. I also tried Carmack's wacky Newton-Rhapson approximation
// which ran even better than the hardware,
// at 4.3ns/float, although with an error of 1 in 210 (which is too much for my purposes).

// The doozy was when I tried the SSE op for reciprocal square root,
// and then used a multiply to get the square root ( x * 1/vx = vx ).
// Even though this takes two dependent operations,
// it was the fastest solution by far, at 1.24ns/float and accurate to 2-14:

inline void SSESqrt_Recip_Times_X( float * restrict pOut, float * restrict pIn )
{
   __m128 in = _mm_load_ss( pIn );
   _mm_store_ss( pOut, _mm_mul_ss( in, _mm_rsqrt_ss( in ) ) );
   // compiles to movss, movaps, rsqrtss, mulss, movss
}
