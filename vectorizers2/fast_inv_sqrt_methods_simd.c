// Suppose that it is necessary to
// compute reciprocal or reciprocal square root for packed floating point data.

__m128 recip_float4_ieee(__m128 x) { return _mm_div_ps(_mm_set1_ps(1.0f), x); }
__m128 rsqrt_float4_ieee(__m128 x) { return _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(x)); }

__m128 recip_float4_half(__m128 x) { return _mm_rcp_ps(x); }
__m128 rsqrt_float4_half(__m128 x) { return _mm_rsqrt_ps(x); }

// Here are the versions for single-precision float numbers with one NR iteration:

__m128 recip_float4_single(__m128 x) {
  __m128 res = _mm_rcp_ps(x);
  __m128 muls = _mm_mul_ps(x, _mm_mul_ps(res, res));
  return res =  _mm_sub_ps(_mm_add_ps(res, res), muls);
}
__m128 rsqrt_float4_single(__m128 x) {
  __m128 three = _mm_set1_ps(3.0f), half = _mm_set1_ps(0.5f);
  __m128 res = _mm_rsqrt_ps(x);
  __m128 muls = _mm_mul_ps(_mm_mul_ps(x, res), res);
  return res = _mm_mul_ps(_mm_mul_ps(half, res), _mm_sub_ps(three, muls));
}

 # Haswell
 // execute (aka dispatch) on cycle 1, results ready on cycle 6
 nr = _mm_rsqrt_ps( x );

 // both of these execute on cycle 6, results ready on cycle 11
 xnr = _mm_mul_ps( x, nr );         // dep on nr
 half_nr = _mm_mul_ps( half, nr );  // dep on nr

 // can execute on cycle 11, result ready on cycle 16
 muls = _mm_mul_ps( xnr , nr );     // dep on xnr

 // can execute on cycle 16, result ready on cycle 19
 three_minus_muls = _mm_sub_ps( three, muls );  // dep on muls

 // can execute on cycle 19, result ready on cycle 24
 result = _mm_mul_ps( half_nr, three_minus_muls ); // dep on three_minus_muls
// result is an approximation of 1/sqrt(x), with ~22 to 23 bits of precision in the mantissa.

#haswell2
// vrsqrtps ymm has higher latency
// execute on cycle 1, results ready on cycle 8
nr = _mm256_rsqrt_ps( x );

// both of can execute on cycle 8, results ready on cycle 13
xnr = _mm256_mul_ps( x, nr );         // dep on nr
half_nr = _mm256_mul_ps( half, nr );  // dep on nr

// can execute on cycle 13, result ready on cycle 18
three_minus_muls = _mm256_fnmadd_ps( xnr, nr, three );  // -(xnr*nr) + 3

// can execute on cycle 18, result ready on cycle 23
result = _mm256_mul_ps( half_nr, three_minus_muls ); // dep on three_minus_muls
