// C(m, n) = A(m, k) * B(k, n)

//// native impl
for (int i = 0; i < m; i++) {
  for (int j = 0; j < n; j++) {
    for (int p = 0; p < k; p++) {
      C(i, j) += A(i, p) * B(p, j);
    }
  }
}

/// Compute a RAxRB block of C using a vectorized dot product, where RA is the
/// number of registers to load from matrix A, and RB is the number of registers
/// to load from matrix B.
template <unsigned regsA, unsigned regsB>
void matmul_dot_inner(int k, const float *a, int lda, const float *b, int ldb,
                      float *c, int ldc) {
  float8 csum[regsA][regsB] = {{0.0}};
  for (int p = 0; p < k; p++) {

    // Perform the DOT product.
    for (int bi = 0; bi < regsB; bi++) {
      float8 bb = LoadFloat8(&B(p, bi * 8));
      for (int ai = 0; ai < regsA; ai++) {
        float8 aa = BroadcastFloat8(A(ai, p));
        csum[ai][bi] += aa * bb;
      }
    }
  }

  // Accumulate the results into C.
  for (int ai = 0; ai < regsA; ai++) {
    for (int bi = 0; bi < regsB; bi++) {
      AdduFloat8(&C(ai, bi * 8), csum[ai][bi]);
    }
  }
}
