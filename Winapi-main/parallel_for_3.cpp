#include "par_for.h"

int main() {
//replace -
for(unsigned i = 0; i < 10; ++i){
    std::cout << i << std::endl;
}

//with -
//method 1:
pl::thread_par_for(0, 10, [&](unsigned i){
            std::cout << i << std::endl;   //do something here with the index i
        });   //changing the end to },false); will make the loop sequantial

//or method 2:
pl::async_par_for(0, 10, [&](unsigned i){
            std::cout << i << std::endl;   //do something here with the index i
        });   //changing the end to },false); will make the loop sequantial

return 0;
}


------------------
#include <thread>
#include <vector>
#include <functional>
#include <future>

using namespace std;

namespace pl{

    void thread_par_for(unsigned start, unsigned end, function<void(unsigned i)> fn, bool par = true){

        //internal loop
        auto int_fn = [&fn](unsigned int_start, unsigned seg_size){
            for (unsigned j = int_start; j < int_start+seg_size; j++){
                fn(j);
            }
        };

        //sequenced for
        if(!par){
            return int_fn(start, end);
        }

        //get number of threads
        unsigned nb_threads_hint = thread::hardware_concurrency();
        unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

        //calculate segments
        unsigned total_length = end - start;
        unsigned seg = total_length/nb_threads;
        unsigned last_seg = seg + total_length%nb_threads;

        //launch threads - parallel for
        auto threads_vec = vector<thread>();
        threads_vec.reserve(nb_threads);
        for(int k = 0; k < nb_threads-1; ++k){
            unsigned current_start = seg*k;
            threads_vec.emplace_back(thread(int_fn, current_start, seg));
        }
        {
            unsigned current_start = seg*(nb_threads-1);
            threads_vec.emplace_back(thread(int_fn, current_start, last_seg));
        }
        for (auto& th : threads_vec){
            th.join();
        }
    }




    void async_par_for(unsigned start, unsigned end, function<void(unsigned i)> fn, bool par = true){

        //internal loop
        auto int_fn = [&fn](unsigned int_start, unsigned seg_size){
            for (unsigned j = int_start; j < int_start+seg_size; j++){
                fn(j);
            }
        };

        //sequenced for
        if(!par){
            return int_fn(start, end);
        }

        //get number of threads
        unsigned nb_threads_hint = thread::hardware_concurrency();
        unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

        //calculate segments
        unsigned total_length = end - start;
        unsigned seg = total_length/nb_threads;
        unsigned last_seg = seg + total_length%nb_threads;

        //launch threads - parallel for
        auto fut_vec = vector<future<void>>();
        fut_vec.reserve(nb_threads);
        for(int k = 0; k < nb_threads-1; ++k){
            unsigned current_start = seg*k;
            fut_vec.emplace_back(async(int_fn, current_start, seg));
        }
        {
            unsigned current_start = seg*(nb_threads-1);
            fut_vec.emplace_back(async(launch::async, int_fn, current_start, last_seg));
        }
        for (auto& th : fut_vec){
            th.get();
        }
    }
}
