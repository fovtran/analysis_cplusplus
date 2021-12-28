using System;
using System.Threading;
using System.Collections.Generic;

namespace T16
{
    class test
    {
        static ManualResetEvent starter = new ManualResetEvent (false);

        //void AppServerMethod { ThreadPool.RegisterWaitForSingleObject (wh, Resume, null, -1, true); }
        public static void T1(  )
        {
            ThreadPool.RegisterWaitForSingleObject (starter, Go, "hello", -1, true);
            Thread.Sleep (5000);
            Console.WriteLine ("Signaling worker...");
            starter.Set(  );
            Console.ReadLine(  );
        }

        public static void Go (object data, bool timedOut)
        {
            Console.WriteLine ("Started " + data);
            // Perform task...
        }
    }
}