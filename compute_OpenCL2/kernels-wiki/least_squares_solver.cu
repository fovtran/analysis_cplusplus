__global__ 
void least_squares_kernel(int factors, int user_count, int item_count, float * X,
                          const float * Y, const float * YtY,
                          const int * indptr, const int * indices, const float * data,
                          int cg_steps) {
    // Ap/r/p are vectors for CG update - use dynamic shared memory to store
    // https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
    extern __shared__ float shared_memory[];
    float * Ap = &shared_memory[0];
    float * r = &shared_memory[factors];
    float * p = &shared_memory[2*factors];

    // Stride over users in the grid:
    // https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int u = blockIdx.x; u < user_count; u += gridDim.x) {
        float * x = &X[u * factors];

        // calculate residual r = YtCuPu - YtCuY Xu
        float temp = 0;
        for (int i = 0; i < factors; ++i) {
            temp -= x[i] * YtY[i * factors + threadIdx.x];
        }
        for (int index = indptr[u]; index < indptr[u + 1]; ++index) {
            const float * Yi = &Y[indices[index] * factors];
            float confidence = data[index];
            temp += (confidence - (confidence - 1) * dot(Yi, x)) * Yi[threadIdx.x];
        }
        p[threadIdx.x] = r[threadIdx.x] = temp;

        float rsold = dot(r, r);

        for (int it = 0; it < cg_steps; ++it) {
            // calculate Ap = YtCuYp - without actually calculating YtCuY
            Ap[threadIdx.x] = 0;
            for (int i = 0; i < factors; ++i) {
                Ap[threadIdx.x] += p[i] * YtY[i * factors + threadIdx.x];
            }
            for (int index = indptr[u]; index < indptr[u + 1]; ++index) {
                const float * Yi = &Y[indices[index] * factors];
                Ap[threadIdx.x] += (data[index] - 1) * dot(Yi, p) * Yi[threadIdx.x];
            }

            // standard CG update
            float alpha = rsold / dot(p, Ap);
            x[threadIdx.x] += alpha * p[threadIdx.x];
            r[threadIdx.x] -= alpha * Ap[threadIdx.x];
            float rsnew = dot(r, r);
            p[threadIdx.x] = r[threadIdx.x] + (rsnew/rsold) * p[threadIdx.x];
            rsold = rsnew;
        }
    }
}

void CudaLeastSquaresSolver::least_squares(const CudaCSRMatrix & Cui,
                                           CudaDenseMatrix * X,
                                           const CudaDenseMatrix & Y,
                                           float regularization,
                                           int cg_steps) const {
    int item_count = Y.rows, user_count = X->rows, factors = X->cols;
    if (X->cols != Y.cols) throw invalid_argument("X and Y should have the same number of columns");
    if (X->cols != YtY.cols) throw invalid_argument("Columns of X don't match number of factors");
    if (Cui.rows != X->rows) throw invalid_argument("Dimensionality mismatch between Cui and X");
    if (Cui.cols != Y.rows) throw invalid_argument("Dimensionality mismatch between Cui and Y");

    // calculate YtY: note this expects col-major (and we have row-major basically)
    // so that we're inverting the CUBLAS_OP_T/CU_BLAS_OP_N ordering to overcome
    // this (like calculate YYt instead of YtY)
    float alpha = 1.0, beta = 0.;
    CHECK_CUBLAS(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             factors, factors, item_count,
                             &alpha,
                             Y.data, factors,
                             Y.data, factors,
                             &beta,
                             YtY.data, factors));
    CHECK_CUDA(cudaDeviceSynchronize());

    // regularize the matrix
    l2_regularize_kernel<<<1, factors>>>(factors, regularization, YtY.data);
    CHECK_CUDA(cudaDeviceSynchronize());

    int block_count = 1024;
    int thread_count = factors;
    int shared_memory_size = sizeof(float) * (3 * factors);

    // Update factors for each user
    least_squares_cg_kernel<<<block_count, thread_count, shared_memory_size>>>(
        factors, user_count, item_count,
        X->data, Y.data, YtY.data, Cui.indptr, Cui.indices, Cui.data, cg_steps);

    CHECK_CUDA(cudaDeviceSynchronize());
}