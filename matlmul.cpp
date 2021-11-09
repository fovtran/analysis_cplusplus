// $ nvcc -arch=sm_12 -Xcompiler="-Wall" -Xptxas="-v" -o matmul2 matmul2.cu

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

inline void GPUassert(cudaError_t code, char * file, int line, bool Abort=true)
{
    if (code != 0) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
        if (Abort) exit(code);
    }       
}

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }

__global__ void matrixMul( float* C, float* A, float* B, int wA, int wB, size_t block_size)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * block_size * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = block_size;
    int bBegin = block_size * bx;
    int bStep  = block_size * wB;

    float Csub=0.f;
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) 
    {
        extern __shared__ float smem[];

        smem[ty*block_size+tx] = A[a + wA * ty + tx];
        smem[block_size*block_size+ty*block_size+tx]  = B[b + wB * ty + tx];

        __syncthreads();

        for (int k = 0; k < block_size; ++k)
            Csub += smem[ty*block_size+k] * smem[block_size*block_size+k*block_size+tx] ;

        __syncthreads();
    }

    int c = wB * block_size * by + block_size * bx;
    C[c + wB * ty + tx] = Csub;
}

inline float frand(){
    return (float)rand()/(float)RAND_MAX;
}

void matmul(float *C, const float *A, const float *B,  int wA, int wB)
{
    for(int k=0; k<wB; k++) {
        for(int j=0; j<wB; j++) {
            float dotp = 0.f;
            for(int i=0; i<wA; i++) {
                dotp += A[j*wA+i] * B[i*wB+k];
            }
            C[j*wB+k] = dotp;
        }
    }
}

int main(int argc, char ** argv)
{
    int val = 128;

    if ( argc == 2 ) {
        val = atoi(argv[1]);
    }

    int m = val, n = val, mn = m*n;
    size_t sz = size_t(mn) * sizeof(float);

    srand(time(NULL));

    float * A = new float[mn], * B = new float[mn], * C= new float[mn];
    float * A_, * B_, * C_;

    for(int i=0; i<mn; i++) {
        A[i] = frand(); B[i] = frand();
    }

    GPUerrchk( cudaMalloc((void **)&A_, sz) );
    GPUerrchk( cudaMalloc((void **)&B_, sz) );
    GPUerrchk( cudaMalloc((void **)&C_, sz) );

    GPUerrchk( cudaMemcpy(A_, A, sz, cudaMemcpyHostToDevice) );
    GPUerrchk( cudaMemcpy(B_, B, sz, cudaMemcpyHostToDevice) );

    // Launch configuration
    // Note that the input matrice sizes *must* be a round
    // multiple of blocksize for this code to work correctly.
    const int blocksize=16;
    const int shmsz = size_t(2*blocksize*blocksize) * sizeof(float);
    dim3 block=dim3(blocksize,blocksize), grid = dim3(m/block.x,m/block.y); 

    matrixMul<<<grid,block,shmsz>>>(C_,A_,B_,m,n,blocksize);
    GPUerrchk( cudaPeekAtLastError() );

    GPUerrchk( cudaMemcpy(C, C_, sz, cudaMemcpyDeviceToHost) );

    // Verfication on host
    float * Cref = new float[mn];
    matmul(Cref,A,B,m,n); 
    const float tol = 5e-5f;
    for(int i=0; i<mn; i++) {
        assert(fabs(C[i]-Cref[i])/C[i] < tol);
    }

    GPUerrchk( cudaThreadExit() ); // CUDA 3.2 compatible

    return 0;
}
