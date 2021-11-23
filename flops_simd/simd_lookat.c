__m128 t = _mm_sub_ps(target, position));
__m128 u = up;
__m128 r = vec4::cross(t, u);
u = vec4::cross(r, t);
t = _mm_sub_ps(_mm_setzero_ps(), t);
_MM_TRANSPOSE4_PS(r, u, t, _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f));  //AoS to SoA

//now normalize
__m128 den = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r,r),_mm_mul_ps(u,u)), _mm_mul_ps(t,t));
__m128 norm = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(den));
r= _mm_mul_ps(norm,r); u =_mm_mul_ps(norm,u); t = _mm_mul_ps(norm,t);