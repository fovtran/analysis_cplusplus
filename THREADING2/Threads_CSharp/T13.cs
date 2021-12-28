using System;
using System.Threading;
using System.Collections.Generic;

namespace T13
{
    class Unsafe
    {
        static bool endIsNigh, repented;

        public static void test1(  )
        {
            new Thread (Wait).Start(  );        // Start up the spinning waiter
            Thread.Sleep (1000);              // Give it a second to warm up!

            repented = true;
            endIsNigh = true;
        }

        static void Wait(  )
        {
            while (!endIsNigh);               // Spin until endIsNigh
            Console.Write (repented);
        }
    }

    class ThreadSafe
    {
        // Always use volatile read/write semantics:
        volatile static bool endIsNigh2, repented2;
    
        public static void test2(  )
        {
            new Thread (Wait2).Start(  );        // Start up the spinning waiter
            Thread.Sleep (1000);              // Give it a second to warm up!

            repented2 = true;
            endIsNigh2 = true;
        }

        static void Wait2(  )
        {
            while (!endIsNigh2);               // Spin until endIsNigh
            Console.Write (repented2);
        }
    }

}