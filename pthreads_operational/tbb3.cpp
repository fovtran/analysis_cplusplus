#include "tbb/task_group.h"

using namespace tbb;

int Fib(int n) {
    if( n<2 ) {
        return n;
    } else {
        int x, y;
        task_group g;
        g.run([&]{x=Fib(n-1);}); // spawn a task
        g.run([&]{y=Fib(n-2);}); // spawn another task
        g.wait();                // wait for both tasks to complete
        return x+y;
    }
}
.//!
 //! Get the default number of threads
 //!
 int nDefThreads = tbb::task_scheduler_init::default_num_threads();

 //!
 //! Init the task scheduler with the wanted number of threads
 //!
 tbb::task_scheduler_init init(nDefThreads);
