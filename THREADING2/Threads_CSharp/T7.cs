using System;
using System.Threading;
using System.Collections.Generic;

namespace T7
{
    class ThreadSafe
    {
        static object locker = new object(  );
        static int x;

        static void Increment(  ) { lock (locker) x++; }
        static void Assign(  )    { lock (locker) x = 123; }
    }

    class test
    {
        static List <string> list = new List <string>(  );

        public static void T1()
        {
            Test();
        }

        static void Test(  )
        {
            lock (list)
            {
                list.Add ("Item 1");
            }
        }
    }
}