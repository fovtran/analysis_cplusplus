using System;
using System.Threading;

namespace T3
{
    class test
    {
        public static void T1()
        {
            ThreadPool.QueueUserWorkItem (Go);
            ThreadPool.QueueUserWorkItem (Go, 123);
            Console.ReadLine(  );
        }

    static void Go (object data)
    {
        Console.WriteLine ("Hello from the thread pool! " + data);
    }

}
}