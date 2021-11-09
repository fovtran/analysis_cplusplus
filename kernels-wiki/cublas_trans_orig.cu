// %comspec% /k ""C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"" amd64
// nvcc -arch=compute_20 -code=sm_20 cublas_trans.cu -o cublas_trans.exe -lcublas
// cp -r c:\VS2013\vc\bin\x86_amd64 c:\VS2013\VC\bin\amd64
// rename c:\VS2013\VC\bin\amd64\vcvarsx86_amd64.bat c:\VS2013\vc\bin\amd64\vcvars64.bat
// nvcc x.cu ...   -ccbin "D:\Program Files\Microsoft Visual Studio 11.0\VC\bin"
#include <cublas_v2.h>

//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
#include <cstring>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

int main(void)
{
	int x_rows_num = 4;
	int q_rows_num = 2;
	int dim = 3;

	int N = x_rows_num*dim;
	int M = q_rows_num*dim;

	float x0[12] = {
			0, 1, 2,
			3, 4, 5,
			6, 7, 8,
			9, 10, 11};

	float q0[6]  = {
			3, 4,
			5, 6,
			7, 8 };

	float *x, *q, *x_q_multiplication;
	CHECK( cudaMallocManaged(&x, N*sizeof(float)) );
	CHECK( cudaMallocManaged(&q, M*sizeof(float)) );
	CHECK( cudaMallocManaged(&x_q_multiplication, q_rows_num*x_rows_num*dim) );

	std::memcpy(x, x0,  N*sizeof(float));
	std::memcpy(q, q0,  M*sizeof(float));

	float *q_device;
	cudaMallocManaged(&q_device, M*sizeof(float));
	cudaMemcpy(q_device, q, M*sizeof(float), cudaMemcpyHostToDevice);

	cublasHandle_t handle; cublasCreate(&handle);

	float alpha = 1.f; float beta = 0.f;

	cublasSgemm(handle,
	        CUBLAS_OP_N, CUBLAS_OP_T,
	        q_rows_num, x_rows_num, dim,
	        &alpha, // 1
	        q_device, q_rows_num,
	        x, x_rows_num,
	        &beta, // 0
	        x_q_multiplication, q_rows_num);

	cudaDeviceSynchronize();

	for (int i = 0; i < q_rows_num*x_rows_num; i++) std::cout << x_q_multiplication[i] << " "; std::cout << std::endl;
	// for (int i = 0; i < q_rows_num*x_rows_num; i++) printf("%d: %f", x_q_multiplication[i]);

	cudaFree(x); cudaFree(q); cudaFree(x_q_multiplication); return 0;
}
