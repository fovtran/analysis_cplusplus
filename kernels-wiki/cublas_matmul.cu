#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

int matrixMultiply()
{
  int devID = 0; cudaDeviceProp deviceProp; cudaGetDeviceProperties(&deviceProp, devID);

  int MatrixA_height = 1;
  int MatrixA_width = 3;
  int MatrixB_height = 3;
  int MatrixB_width = 4;

  int MatrixC_width = MatrixB_width;
  int MatrixC_height = MatrixA_height;

  unsigned int MatrixA_size = MatrixA_width * MatrixA_height;
  unsigned int MatrixA_mem_size = sizeof(float) * MatrixA_size;
  float h_MatrixA[3] = { 0.5f, -0.5f, 3.0f};

  unsigned int MatrixB_size = MatrixB_width * MatrixB_height;
  unsigned int MatrixB_mem_size = sizeof(float) * MatrixB_size;
  float h_MatrixB[12] = {
        -0.9f, -0.8f, -0.7f, -0.6f,
        -0.5f, -0.4f, -0.3f, -0.2f,
        -0.1f, 0.0f, 0.0f, 0.0f };

  float h_MatrixC[4];
  unsigned int MatrixC_size = MatrixC_width * MatrixC_height;
  unsigned int MatrixC_mem_size = sizeof(float) * MatrixC_size;

  // allocate device memory
  float *d_MatrixA, *d_MatrixB, *d_MatrixC;
  cudaMalloc((void **) &d_MatrixA, MatrixA_mem_size);
  cudaMalloc((void **) &d_MatrixB, MatrixB_mem_size);
  cudaMalloc((void **) &d_MatrixC, MatrixC_mem_size);

  cudaMemcpy(d_MatrixA, h_MatrixA, MatrixA_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_MatrixB, h_MatrixB, MatrixB_mem_size, cudaMemcpyHostToDevice);

  cublasHandle_t handle; cublasCreate(&handle); cublasStatus_t ret;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  ret = cublasSgemm
  (
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    MatrixA_height, MatrixB_width, MatrixA_width,
    &alpha,
    d_MatrixB, MatrixB_width,
    d_MatrixA, MatrixA_width,
    &beta,
    d_MatrixC, MatrixA_width
  );

  if (ret != CUBLAS_STATUS_SUCCESS) { printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__); return 1; }

  // copy result from device to host
  cudaMemcpy(h_MatrixC, d_MatrixC, MatrixC_mem_size, cudaMemcpyDeviceToHost);

  for(int i = 0; i< MatrixC_size; i++) { printf("%d: %f", i, h_MatrixC[i]); }

  cudaFree(d_MatrixA); cudaFree(d_MatrixB); cudaFree(d_MatrixC); return 0;
}

int main(void)
{
  matrixMultiply();
  cudaDeviceReset(); //getchar();
  return 0;
}
