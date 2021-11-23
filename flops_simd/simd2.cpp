void vec_copy(float *dest, float *src, int len)
{
  float ii;
  #pragma simd
  for (int i = 0, ii = 0.0f; i < len; i++)
  	   dest[i] = src[i] * ii++;
}
