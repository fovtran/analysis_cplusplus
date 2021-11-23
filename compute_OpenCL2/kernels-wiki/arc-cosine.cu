#define ACOS_TESTS (5)
#define ACOS_THREAD_CNT (128)
#define ACOS_CTA_CNT (96)

struct acosParams {
 float *arg;
 float *res;
 int n;
};

__global__ void acos_main (struct acosParams parms)
{
 int i;
 int totalThreads = gridDim.x * blockDim.x;
 int ctaStart = blockDim.x * blockIdx.x;
 for (i = ctaStart + threadIdx.x; i < parms.n; i += totalThreads) {
 parms.res[i] = acosf(parms.arg[i]);
 }
}

int main (int argc, char *argv[])
{
 volatile float acosRef;
 float* acosRes = 0;
 float* acosArg = 0;
 float* arg = 0;
 float* res = 0;
 float t;
 struct acosParams funcParams;
 int errors;
 int i;
 cudaMalloc ((void **)&acosArg, ACOS_TESTS * sizeof(float));
 cudaMalloc ((void **)&acosRes, ACOS_TESTS * sizeof(float));

 arg = (float *) malloc (ACOS_TESTS * sizeof(arg[0]));
 res = (float *) malloc (ACOS_TESTS * sizeof(res[0]));
 cudaMemcpy (acosArg, arg, ACOS_TESTS * sizeof(arg[0]),
 cudaMemcpyHostToDevice);

 funcParams.res = acosRes;
 funcParams.arg = acosArg;
 funcParams.n = opts.n;
 acos_main<<<ACOS_CTA_CNT,ACOS_THREAD_CNT>>>(funcParams);
 cudaMemcpy (res, acosRes, ACOS_TESTS * sizeof(res[0]),
 cudaMemcpyDeviceToHost);

 return 0;
 }