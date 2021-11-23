__shared__ int N_per_data;  // loaded from host
__shared__ float ** data;  //loaded from host
data = new float*[num_threads_in_warp];
for (int j = 0; j < num_threads_in_warp; ++j)
     data[j] = new float[N_per_data[j]];

// the values of jagged matrix "data" are loaded from host.


__shared__  float **max_data = new float*[num_threads_in_warp];
for (int j = 0; j < num_threads_in_warp; ++j)
     max_data[j] = new float[N_per_data[j]];

for (uint j = 0; j <  N_per_data[threadIdx.x]; ++j)
{
   const float a = f(data[threadIdx.x][j]);
   const float b = g(data[threadIdx.x][j]);
   const float c = h(data[threadIdx.x][j]);

  const int cond_a = (a > b)  &&  (a > c);
  const int cond_b = (b > a)  && (b > c);
  const int cond_c = (c > a)  && (c > b);

  // avoid if-statements.  question (1) and (2)
  max_data[threadIdx.x][j] =   conda_a * a  +  cond_b * b  +  cond_c * c; 
}



 // Question (3):
// No "syncthreads"  necessary in next line:

// access data of your mate at some magic positions (assume it exists):
float my_neighbors_max_at_7 = max_data[threadIdx.x + pow(-1,(threadIdx.x % 2) == 1) ][7]; 