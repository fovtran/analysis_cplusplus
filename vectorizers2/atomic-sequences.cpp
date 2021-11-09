// >cl /EHsc complex.cpp /std:c++latest
#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define _USE_MATH_DEFINES
#include <vector>
#include <math.h>
#include <cmath>
#include <complex>
#include <complex.h>

using namespace std;

#define __float double

atomic<uint64_t> seq; // seqlock representation
int data1, data2;     // this data will be protected by seq

T reader() {
    int r1, r2;
    unsigned seq0, seq1;
    while (true) {
        seq0 = seq;
        r1 = data1; // INCORRECT! Data Race!
        r2 = data2; // INCORRECT!
        seq1 = seq;

        // if the lock didn't change while I was reading, and
        // the lock wasn't held while I was reading, then my
        // reads should be valid
        if (seq0 == seq1 && !(seq0 & 1))
            break;
    }
    use(r1, r2);
}

void writer(int new_data1, int new_data2) {
    unsigned seq0 = seq;
    while (true) {
        if ((!(seq0 & 1)) && seq.compare_exchange_weak(seq0, seq0 + 1))
            break; // atomically moving the lock from even to odd is an acquire
    }
    data1 = new_data1;
    data2 = new_data2;
    seq = seq0 + 2; // release the lock by increasing its value to even
}

As unintuitive as it seams at first, data1 and data2 need to be atomic<>. If they are not atomic, then they could be read (in reader()) at the exact same time as they are written (in writer()). According to the C++ memory model, this is a race even if reader() never actually uses the data. In addition, if they are not atomic, then the compiler can cache the first read of each value in a register. Obviously you wouldn't want that... you want to re-read in each iteration of the while loop in reader().

It is also not sufficient to make them atomic<> and access them with memory_order_relaxed. The reason for this is that the reads of seq (in reader()) only have acquire semantics. In simple terms, if X and Y are memory accesses, X precedes Y, X is not an acquire or release, and Y is an acquire, then the compiler can reorder Y before X. If Y was the second read of seq, and X was a read of data, such a reordering would break the lock implementation.

The paper gives a few solutions. The one with the best performance today is probably the one that uses an atomic_thread_fence with memory_order_relaxed before the second read of the seqlock. In the paper, it's Figure 6. I'm not reproducing the code here, because anyone who has read this far really ought to read the paper. It is more precise and complete than this post.

The last issue is that it might be unnatural to make the data variables atomic. If you can't in your code, then you need to be very careful, because casting from non-atomic to atomic is only legal for primitive types. C++20 is supposed to add atomic_ref<>, which will make this problem easier to resolve.
