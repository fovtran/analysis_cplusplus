// c:\VS2013\VC\vcvarsall.bat
// nvcc -arch=compute_20 -code=sm_20 cublas_trans.cu -o cublas_trans.exe -lcublas -g -G -run -Xcompiler -openmp

// set NVCC_FLAGS=-arch=compute_20 --gpu-code=compute_20,sm_20,sm_21,sm_30,sm_50 -lcublas -lcurand -lcufft -g -G --relocatable-device-code=true -std=c++11 -lcudart -O3 -run -Xcompiler -openmp
// nvcc %NVCC_FLAGS% cublas_trans.cu -o cublas_trans.exe
// set DXSDK_DIR=D:\DATASA\LIB2\DirectX_June2010

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>
#include <math_constants.h>
#include <vector_functions.h>

// #define arch __CUDA_ARCH__

void printHostMatrix(int nl, int nc, float *h_s){
		printf("\nC=[");
	    for(int j = 0; j < nl ; j++) {
				printf("\n\t[ ");
	        for(int i = 0; i < (nc) ; i++){
	            int idx = j*nc + i;
	            printf("\t\t%.2f ", h_s[idx]);
	        }
	        printf("\t]\n");
	    }
			printf("  ]\n");
}

int matrixMultiply()
{
	#ifdef _OPENMP
    printf("OpenMP Enabled\n");
    printf("Will use: %d threads of %d available\n",
            omp_get_num_procs(), omp_get_max_threads());
		// printf("CUDA Compute v %f", arch);
  #endif

	int devID = 0; cudaDeviceProp deviceProp; cudaGetDeviceProperties(&deviceProp, devID);
	cublasHandle_t handle; cublasCreate(&handle); // cublasStatus_t ret;

	float *A, *B, *C, *_C;
	float alpha = 1.0f; float beta = 0.0f;

	float A0[6]  = {
			1,2,3,
			4,5,6 };

	float B0[12] = {
			7.0f, 8.0f,
			9.0f, 10.0f,
			11.0f, 12.0f };

	int A_height = 2;
  int A_width = 3;
	int B_height = 3;
  int B_width = 2;

  int C_width = B_width;
  int C_height = A_height;

  unsigned int A_size = A_width * A_height;
  unsigned int A_mem_size = sizeof(float) * A_size;

	unsigned int B_size = B_width * B_height;
  unsigned int B_mem_size = sizeof(float) * B_size;

	unsigned int C_size = C_width * C_height;
  unsigned int C_mem_size = sizeof(float) * C_size;

	cudaMalloc((void**) &A, A_mem_size);
	cudaMalloc((void**) &B, B_mem_size);
	cudaMalloc((void**) &_C, C_mem_size);
	C = (float*) malloc(C_mem_size);

	cudaMemcpy(A, A0, A_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(B, B0, B_mem_size, cudaMemcpyHostToDevice);

	cublasSgemm(handle,
	        CUBLAS_OP_N, CUBLAS_OP_N,
	        A_height, B_width, A_width,
	        &alpha,
	        B, B_width,
	        A, A_width,
	        &beta,
	        _C, C_width);

	cudaMemcpy(C, _C, C_mem_size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < C_size; i++) std::cout << i << ": " << C[i] << "  "; std::cout << std::endl;
	for (int i = 0; i < C_size; i++) printf("%d: %f ", i, C[i]);

	printHostMatrix(C_height, C_width, C);
	cudaDeviceSynchronize(); cudaFree(B); cudaFree(A); cudaFree(C); cudaFree(_C); return 0;
}

int main(void)
{
	matrixMultiply();
	return 0;
}
