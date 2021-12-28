#include <iostream>
#include <cstdlib>
#include <pthread.h>

using namespace std;

#define NUM_THREADS 2

int arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

void *splitLoop(void *threadid)
{
   long tid;
   tid = (long)threadid;
   //cout << "Hello World! Thread ID, " << tid << endl;
   int start = (tid * 5);
   int end = start + 5;
   for(int i = start;i < end;i++){
      cout << arr[i] << " ";
   }
   cout << endl;
   pthread_exit(NULL);
}

int main ()
{
   pthread_t threads[NUM_THREADS];
   int rc;
   int i;
   for( i=0; i < NUM_THREADS; i++ ){
      cout << "main() : creating thread, " << i << endl;
      rc = pthread_create(&threads[i], NULL,
                          splitLoop, (void *)i);
      if (rc){
         cout << "Error:unable to create thread," << rc << endl;
         exit(-1);
      }
   }
   pthread_exit(NULL);
}


int main() {
    srand(time(NULL)); // seed
    const int N1 = 1000;
    const int N2 = 100000;
    int n = 0;
    int c = 0;
    Concurrency::critical_section cs;
    // it is better that N2 >> N1 for better performance
    Concurrency::parallel_for(0, N1, [&](int i) {
        int t = monte_carlo_count_pi(N2);
        cs.lock(); // race condition
        n += N2;   // total sampling points
        c += t;    // points fall in the circle
        cs.unlock();
    });
    cout < < "pi ~= " << setprecision(9) << (double)c / n * 4.0 << endl;
    return 0;
}
