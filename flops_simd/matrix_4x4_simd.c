I am trying to find the most efficient implementation of 4x4 matrix (M) multiplication 
with a vector (u) using SSE. I mean Mu = v.

As far as I understand there are two primary ways to go about this:

    method 1) v1 = dot(row1, u), v2 = dot(row2, u), v3 = dot(row3, u), v4 = dot(row4, u)
    method 2) v = u1 col1 + u2 col2 + u3 col3 + u4 col4.
    
    
    __m128 m4x4v_colSSE(const __m128 cols[4], const __m128 v) {
      __m128 u1 = _mm_shuffle_ps(v,v, _MM_SHUFFLE(0,0,0,0));
      __m128 u2 = _mm_shuffle_ps(v,v, _MM_SHUFFLE(1,1,1,1));
      __m128 u3 = _mm_shuffle_ps(v,v, _MM_SHUFFLE(2,2,2,2));
      __m128 u4 = _mm_shuffle_ps(v,v, _MM_SHUFFLE(3,3,3,3));
    
      __m128 prod1 = _mm_mul_ps(u1, cols[0]);
      __m128 prod2 = _mm_mul_ps(u2, cols[1]);
      __m128 prod3 = _mm_mul_ps(u3, cols[2]);
      __m128 prod4 = _mm_mul_ps(u4, cols[3]);
    
      return _mm_add_ps(_mm_add_ps(prod1, prod2), _mm_add_ps(prod3, prod4));
    }
    
    __m128 m4x4v_rowSSE3(const __m128 rows[4], const __m128 v) {
      __m128 prod1 = _mm_mul_ps(rows[0], v);
      __m128 prod2 = _mm_mul_ps(rows[1], v);
      __m128 prod3 = _mm_mul_ps(rows[2], v);
      __m128 prod4 = _mm_mul_ps(rows[3], v);
    
      return _mm_hadd_ps(_mm_hadd_ps(prod1, prod2), _mm_hadd_ps(prod3, prod4));
    }
    
    __m128 m4x4v_rowSSE4(const __m128 rows[4], const __m128 v) {
      __m128 prod1 = _mm_dp_ps (rows[0], v, 0xFF);
      __m128 prod2 = _mm_dp_ps (rows[1], v, 0xFF);
      __m128 prod3 = _mm_dp_ps (rows[2], v, 0xFF);
      __m128 prod4 = _mm_dp_ps (rows[3], v, 0xFF);
    
      return _mm_shuffle_ps(_mm_movelh_ps(prod1, prod2), _mm_movelh_ps(prod3, prod4),  _MM_SHUFFLE(2, 0, 2, 0));
}  