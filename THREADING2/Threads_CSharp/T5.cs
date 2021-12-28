using System;
using System.Threading;

namespace T5
{
    class test
    {
        delegate int WorkInvoker (string text);
        
        public static void T1()
        {
            WorkInvoker method = Work;
            IAsyncResult cookie = method.BeginInvoke ("test", null, null);
            //
            // ... here's where we can do other work in parallel...
            //
            int result = method.EndInvoke (cookie);
            Console.WriteLine ("String length is: " + result);
        }

        static int Work (string s) { return s.Length; }
}
}