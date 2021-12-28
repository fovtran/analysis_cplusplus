using System;
using System.Threading;

namespace T9
{
    class test
    {
        static Semaphore s = new Semaphore (3, 3);   // Available=3; Capacity=3

        static void T1(  )
            {
                for (int i = 1; i <= 5; i++) new Thread (Enter).Start (i);
            }

        static void Enter (object id)
        {
            Console.WriteLine (id + " wants to enter");s.WaitOne(  );
            Console.WriteLine (id + " is in!");           // Only three threads
            Thread.Sleep (1000 * (int) id);               // can be here at
            Console.WriteLine (id + " is leaving");       // a time.
            s.Release(  );
        }
    }
}