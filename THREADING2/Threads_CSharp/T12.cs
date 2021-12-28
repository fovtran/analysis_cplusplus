using System;
using System.Threading;
using System.Collections.Generic;

namespace T12
{
    class Atomicity        // This assumes we're running on a 32-bit CPU.
    {
        static int x, y;
        static long z;
        static long sum;

        static void Test(  )
        {
            long myLocal;
            x = 3;             // Atomic
            z = 3;             // Nonatomic (z is 64 bits)
            myLocal = z;       // Nonatomic (z is 64 bits)
            y += x;            // Nonatomic (read AND write operation)
            x++;               // Nonatomic (read AND write operation)
        }

        static void test2()
        {                                                               // sum
            // Simple increment/decrement operations:
            Interlocked.Increment (ref sum);                              // 1
            Interlocked.Decrement (ref sum);                              // 0
            // Add/subtract a value:
            Interlocked.Add (ref sum, 3);                                 // 3

            // Read a 64-bit field:
            Console.WriteLine (Interlocked.Read (ref sum));               // 3

            // Write a 64-bit field while reading previous value:
            // (This prints "3" while updating sum to 10)
            Console.WriteLine (Interlocked.Exchange (ref sum, 10));       // 10

            // Update a field only if it matches a certain value (10):
            Interlocked.CompareExchange (ref sum, 123, 10);               // 123
        }
    }
}